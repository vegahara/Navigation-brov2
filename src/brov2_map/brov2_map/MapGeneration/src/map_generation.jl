module MapGenerationFunctions

using StaticArrays
using LinearAlgebra

import NearestNeighbors
import Distances
import Statistics
import Distributions
import Images
import ImageFiltering


# Struct to represent the polar coordinates of each of the four corners, referenced to a local frame.
# First element contains the range and second theta.  
struct CellLocal
    tl_corner::SVector{2,Float64}
    tr_corner::SVector{2,Float64}
    br_corner::SVector{2,Float64}
    bl_corner::SVector{2,Float64}
end

mutable struct Swath
    data_port
    data_stb
    odom::SVector{5,Float64} #(x,y,roll,pitch,yaw)
    altitude::Float64
end

function generate_map(n_rows, n_colums, n_bins, map_resolution, map_origin_x, map_origin_y, swaths, sonar_range, sonar_horizontal_beam_spread, swath_ground_resolution)

    map_origin = SVector{2,Float64}(map_origin_x, map_origin_y)
    probability_map = fill(1.0, (n_rows, n_colums))
    echo_intensity_map = fill(NaN, (n_rows, n_colums))
    range_map = fill(NaN, (n_rows, n_colums))
    altitude_map = fill(NaN, (n_rows, n_colums))
    observed_swath_map = fill([], (n_rows, n_colums))
    
    swath_locals = [Swath(
        swath.data_port, 
        swath.data_stb, 
        SVector{5,Float64}(
            swath.odom[1]+swath.altitude*sin(swath.odom[4])*cos(swath.odom[5]), # Pitch correction
            swath.odom[2]+swath.altitude*sin(swath.odom[4])*sin(swath.odom[5]), # Pitch correction
            swath.odom[3],
            swath.odom[4],
            swath.odom[5]),
        float(swath.altitude)) 
        for swath in swaths]

    min_intensity = 100.0
    max_intensity = 0.0

    for swath in swath_locals
        for bin in swath.data_port
            if isnan(bin)
                continue
            end
            min_intensity = min(min_intensity, bin)
            max_intensity = max(max_intensity, bin)
        end
        for bin in swath.data_stb
            if isnan(bin)
                continue
            end
            min_intensity = min(min_intensity, bin)
            max_intensity = max(max_intensity, bin)
        end
    end

    for row=1:n_rows, col=1:n_colums
        cell_global = get_cell_global_coordinates(row,col,map_resolution,map_origin)
        echo_intensities = []
        swath_probabilities = []
        altitudes = []
        observed_swaths = []
        probability = 1.0
        cell_range = NaN
        cell_observed = false

        for (swath, swath_number) in zip(swath_locals, 1:length(swath_locals))

            cell_l = get_cell_coordinates(cell_global, swath.odom, map_resolution)

            prob_swath = get_cell_probability_gaussian(cell_l, sonar_range, sonar_horizontal_beam_spread)

            if prob_swath >= 0.1

                echo_intensity_swath = get_cell_echo_intensity(cell_l, swath, swath_ground_resolution, n_bins)
                
                if !isnan(echo_intensity_swath)
                    cell_observed = true

                    push!(echo_intensities, echo_intensity_swath)
                    push!(swath_probabilities, prob_swath)
                    push!(altitudes, swath.altitude)
                    push!(observed_swaths, swath_number - 1) # 0 indexed

                    cell_range = minimum(cell_range->isnan(cell_range) ? -Inf : cell_range, get_cell_range(cell_l))
                
                    probability *= (1 - prob_swath)
                end

            end
        end

        if cell_observed
            normalized_swath_probabilities = swath_probabilities ./ sum(swath_probabilities)
            probability_map[row,col] = probability
            echo_intensity_map[row,col] = sum(normalized_swath_probabilities .* echo_intensities)
            range_map[row,col] = cell_range
            altitude_map[row,col] = sum(altitudes) / length(altitudes)
            observed_swath_map[row,col] = observed_swaths
        end
    end

    k = 4;
    variance_ceiling = 0.05
    # variance_ceiling = 5
    max_distance = 0.2

    echo_intensity_map, intensity_variance = knn(
        n_rows, n_colums, echo_intensity_map, 
        map_resolution, k, variance_ceiling, max_distance
    )
    
    echo_intensity_map = speckle_reducing_bilateral_filter(echo_intensity_map, 0.1)
    echo_intensity_map = speckle_reducing_bilateral_filter(echo_intensity_map, 0.3)
    echo_intensity_map = speckle_reducing_bilateral_filter(echo_intensity_map, 0.5)

    # echo_intensity_map = bilateral_filter(echo_intensity_map, 0.1, 0.5)
    # echo_intensity_map = bilateral_filter(echo_intensity_map, 0.3, 0.7)
    # echo_intensity_map = bilateral_filter(echo_intensity_map, 0.5, 0.9)
    
    return echo_intensity_map, probability_map, observed_swath_map, range_map
    # return intensity_variance, probability_map
    # return echo_intensity_map, intensity_variance
end

function get_cell_global_coordinates(row, colum, map_resolution, map_origin)
    cell_global_x = map_origin[1] - (row - 1) * map_resolution
    cell_global_y = map_origin[2] + (colum - 1) * map_resolution 
    
    return SVector(cell_global_x,cell_global_y)
end

function get_cell_coordinates(cell_global, local_frame, map_resolution)

    v1 = cell_global - local_frame[1:2]
    v2 = v1 + SVector(0, map_resolution)
    v3 = v2 + SVector(-map_resolution, 0)
    v4 = v1 + SVector(-map_resolution, 0)

    n1 = norm(v1)
    n2 = norm(v2)
    n3 = norm(v3)
    n4 = norm(v4)

    a = rem2pi.(atan.([v1[2], v2[2], v3[2], v4[2]], [v1[1], v2[1], v3[1], v4[1]]) .- local_frame[5], RoundDown)

    return CellLocal(SVector(n1, a[1]), SVector(n2, a[2]), SVector(n3, a[3]), SVector(n4, a[4]))
end


function is_cell_observed(cell_l, sonar_range, sonar_horizontal_beam_spread)

    corners = [cell_l.tl_corner, cell_l.tr_corner, cell_l.br_corner, cell_l.bl_corner]

    # Correct based on left or right swath
    corr = pi/2 + (corners[1][2] > pi)*pi

    min_r = corners[1][1]
    min_theta = corners[1][2] - corr
    max_theta = min_theta
    abs_min_theta = abs(min_theta)

    for i = 2:4
        min_r = min(min_r, corners[i][1])
        new_theta = corners[i][2] - corr
        abs_min_theta = min(abs_min_theta, abs(new_theta))
        min_theta = min(min_theta, new_theta)
        max_theta = max(max_theta, new_theta)
    end

    # (max_theta > 0 && min_theta < 0) means that there is one corner on each side of the acoustic axis 
    # hence, the cell is observed
    # ((max_theta - min_theta) < pi) is to fix wrap around bug

    return (min_r < sonar_range) && ((abs_min_theta < sonar_horizontal_beam_spread) || (max_theta > 0 && min_theta < 0 && (max_theta - min_theta) < pi))
end 

# With range checking
function get_cell_probability_gaussian(cell_l, sonar_range, sonar_horizontal_beam_spread)
    
    corners = [cell_l.tl_corner, cell_l.tr_corner, cell_l.br_corner, cell_l.bl_corner]

    # Correct based on left or right swath
    corr = pi/2 + (corners[1][2] > pi)*pi

    max_r = corners[1][1]
    min_theta = corners[1][2] - corr
    max_theta = min_theta
 
    for i = 2:4
        max_r = max(max_r, corners[i][1])
        new_theta = corners[i][2] - corr
        min_theta = min(min_theta, new_theta)
        max_theta = max(max_theta, new_theta)
    end

    if (max_r < sonar_range) && ((max_theta - min_theta) < pi)
        d = Distributions.Normal(0.0, sonar_horizontal_beam_spread)
        return Distributions.cdf(d, max_theta) - Distributions.cdf(d, min_theta)
    else
        return 0.0
    end
end
    

function get_cell_probability_uniform(cell_l, sonar_horizontal_beam_spread)

    corners = [cell_l.tl_corner, cell_l.tr_corner, cell_l.br_corner, cell_l.bl_corner]

    min_theta = corners[1][2]
    max_theta = corners[1][2]

    for i = 2:4
        min_theta = min(min_theta, corners[i][2])
        max_theta = max(max_theta, corners[i][2])
    end

    if (max_theta - min_theta) > sonar_horizontal_beam_spread
        # Pixel is covering the whole beam and we cant have more than 1 in probability
        return 1
    else
        return (max_theta - min_theta) / sonar_horizontal_beam_spread 
    end
end


function get_cell_probability_gaussian(cell_l, sonar_horizontal_beam_spread)

    corners = [cell_l.tl_corner, cell_l.tr_corner, cell_l.br_corner, cell_l.bl_corner]

    # Correct based on left or right swath
    corr = pi/2 + (corners[1][2] > pi)*pi

    min_theta = corners[1][2] - corr
    max_theta = min_theta

    for i = 2:4
        new_theta = corners[i][2] - corr
        min_theta = min(min_theta, new_theta)
        max_theta = max(max_theta, new_theta)
    end
    
    d = Distributions.Normal(0.0, sonar_horizontal_beam_spread)

    return Distributions.cdf(d, max_theta) - Distributions.cdf(d, min_theta)
end


function get_cell_echo_intensity(cell_l, swath, swath_resolution, n_bins)

    pixel_intensity = 0
    valid_corners = 0

    for f in fieldnames(typeof(cell_l))
        corner = getfield(cell_l,f)

        # Interpolate
        lower_index = floor(Int64, corner[1]/swath_resolution)
        higher_index = ceil(Int64, corner[1]/swath_resolution)
        w1 = corner[1]/swath_resolution - lower_index
        w2 = 1 - w1

        if (lower_index == 0)
            continue
        end

        # The corner is outside the swath range or not measured
        if (higher_index > n_bins) || (lower_index == 0)
            continue
        end

        if corner[2] > pi
            measure_intensity = w1*swath.data_stb[higher_index] + w2*swath.data_stb[lower_index]
        else
            measure_intensity = w1*reverse(swath.data_port)[higher_index] + w2*reverse(swath.data_port)[lower_index]
        end

        # Do not use corner if it evaluate to NaN
        if isnan(measure_intensity)
            continue
        end

        pixel_intensity += measure_intensity
        valid_corners += 1

    end

    return pixel_intensity/valid_corners
end

function get_cell_range(cell_l)

    sum = 0

    for f in fieldnames(typeof(cell_l))
        sum += getfield(cell_l,f)[1]
    end

    return sum/4

end

function knn(n_rows, n_colums, echo_map, map_resolution, k, variance_ceiling, max_distance)
    
    # Make data vectors to use in kdtree
    cell_coordinates = SVector{2,Float64}[]
    intensity_values = []

    for row=1:n_rows, col=1:n_colums
        if !isnan(echo_map[row,col])
            push!(cell_coordinates,SVector{2,Float64}(row*map_resolution,col*map_resolution))
            push!(intensity_values,echo_map[row,col])
        end
    end

    intensity_mean = fill(NaN,n_rows,n_colums)
    intensity_variance = fill(NaN,n_rows,n_colums)

    kdtree = NearestNeighbors.KDTree(cell_coordinates, Distances.Euclidean())

    for row=1:n_rows, col=1:n_colums
        if !isnan(echo_map[row,col])
            intensity_mean[row,col] = echo_map[row,col] 
            continue
        end

        idx, dist = NearestNeighbors.knn(
            kdtree,
            SVector{2,Float64}(row*map_resolution,col*map_resolution),
            k
        )
        svals = [intensity_values[idx[ind]] for ind in 1:k if dist[ind] <= max_distance]

        if svals != []
            var = Statistics.var(svals)
            if var <= variance_ceiling
                intensity_mean[row,col] = Statistics.mean(svals)
                intensity_variance[row,col] = var
            else
                intensity_mean[row,col] = Statistics.quantile!(svals, 10/100)
                intensity_variance[row,col] = variance_ceiling
            end
        end
    end
    
    return (intensity_mean, intensity_variance)
end

function speckle_reducing_bilateral_filter(input_image::Array{T, 2}, sigma_s::Float64) where T<:Number
    output_image = similar(input_image)
    window_size = ceil(Int, 3 * sigma_s)
    for i in 1:size(input_image, 1)
        for j in 1:size(input_image, 2)

            if isnan(input_image[i,j])
                output_image[i,j] = NaN
                continue
            end

            numerator = 0.0
            denominator = 0.0

            for k in max(1, i-window_size):min(size(input_image, 1), i+window_size)
                for l in max(1, j-window_size):min(size(input_image, 2), j+window_size)

                    if isnan(input_image[k,l])
                        continue
                    end

                    spatial_sup = exp(-((k-i)^2 + (l-j)^2)/(2*sigma_s^2))
                    range_sup = (((2 * input_image[i,j]) / (input_image[k,l] ^ 2)) 
                                * exp(-(input_image[i,j]^2)/(input_image[k,l] ^ 2)) )                 
                    weight = spatial_sup * range_sup
                    numerator += input_image[k,l] * weight
                    denominator += weight
                end
            end
            if denominator == 0 # if no valid pixels in the window, output NaN
                output_image[i,j] = NaN
            else
                output_image[i,j] = numerator / denominator
            end
        end
    end
    return output_image
end

function bilateral_filter(input_image::Array{T, 2}, sigma_s::Float64, sigma_r::Float64) where T<:Number
    output_image = similar(input_image)
    window_size = ceil(Int, 3 * sigma_s)
    for i in 1:size(input_image, 1)
        for j in 1:size(input_image, 2)

            if isnan(input_image[i,j])
                output_image[i,j] = NaN
                continue
            end

            numerator = 0.0
            denominator = 0.0

            for k in max(1, i-window_size):min(size(input_image, 1), i+window_size)
                for l in max(1, j-window_size):min(size(input_image, 2), j+window_size)
                    
                    if isnan(input_image[k,l])
                        continue
                    end

                    spatial_dist = sqrt((k-i)^2 + (l-j)^2)
                    range_dist = abs(input_image[i,j] - input_image[k,l])
                    weight = exp(-spatial_dist^2/(2*sigma_s^2) - range_dist^2/(2*sigma_r^2))
                    numerator += input_image[k,l] * weight
                    denominator += weight
                end
            end

            if denominator == 0 # if no valid pixels in the window, output NaN
                output_image[i,j] = NaN
            else
                output_image[i,j] = numerator / denominator
            end
        end
    end
    return output_image
end

end
