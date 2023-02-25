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
    inverse_intensity_map = fill(NaN, (n_rows, n_colums))
    
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

    inverse_swath_locals = [Swath(
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

    for inverse_swath in inverse_swath_locals
        inverse_swath.data_port = (max_intensity + min_intensity) .-inverse_swath.data_port
        inverse_swath.data_stb = (max_intensity + min_intensity) .-inverse_swath.data_stb
    end

    for row=1:n_rows, col=1:n_colums
        cell_global = get_cell_global_coordinates(row,col,map_resolution,map_origin)
        echo_intensities = []
        inverse_intesitites = []
        swath_probabilities = []
        probability = 1.0
        cell_observed = false

        for (swath, inverse_swath) in zip(swath_locals, inverse_swath_locals)

            cell_l = get_cell_coordinates(cell_global, swath.odom, map_resolution)

            prob_swath = get_cell_probability_gaussian(cell_l, sonar_range, sonar_horizontal_beam_spread)

            if prob_swath >= 0.1

                echo_intensity_swath = get_cell_echo_intensity(cell_l, swath, swath_ground_resolution, n_bins)
                inverse_intensity_swath = get_cell_echo_intensity(cell_l, inverse_swath, swath_ground_resolution, n_bins)
                
                if !isnan(echo_intensity_swath)
                    cell_observed = true

                    push!(echo_intensities, echo_intensity_swath)
                    push!(inverse_intesitites, inverse_intensity_swath)
                    push!(swath_probabilities, prob_swath)
                
                    probability *= (1 - prob_swath)
                end

            end
        end

        if cell_observed
            normalized_swath_probabilities = swath_probabilities ./ sum(swath_probabilities)
            probability_map[row,col] = probability
            echo_intensity_map[row,col] = sum(normalized_swath_probabilities .* echo_intensities)
            inverse_intensity_map[row,col] = sum(normalized_swath_probabilities .* inverse_intesitites)
        end

        if col == n_colums
            println(row)
        end
    end

    min_intensity = minimum(inverse_intensity_map[.!isnan.(inverse_intensity_map)])
    max_intensity = maximum(inverse_intensity_map[.!isnan.(inverse_intensity_map)])

    inverse_intensity_map = (min_intensity + max_intensity) .- inverse_intensity_map

    k = 4;
    # variance_ceiling = 0.05
    variance_ceiling = 5
    max_distance = 0.5

    # intensity_mean, intensity_variance, echo_intensity_map = knn(
    #     n_rows, n_colums, echo_intensity_map, 
    #     map_resolution, k, variance_ceiling, max_distance
    # )

    # inverse_intensity_mean, inverse_intensity_variance, inverse_intensity_map = knn(
    #     n_rows, n_colums, inverse_intensity_map, 
    #     map_resolution, k, variance_ceiling, max_distance
    # )
    
    # echo_intensity_map = speckle_reducing_bilateral_filter(n_rows, n_colums, echo_intensity_map, 1.0)
    # echo_intensity_map = speckle_reducing_bilateral_filter(n_rows, n_colums, echo_intensity_map, 0.7)
    # echo_intensity_map = speckle_reducing_bilateral_filter(n_rows, n_colums, echo_intensity_map, 0.5)
    # echo_intensity_map = speckle_reducing_bilateral_filter(n_rows, n_colums, echo_intensity_map, 0.3)
    # echo_intensity_map = speckle_reducing_bilateral_filter(n_rows, n_colums, echo_intensity_map, 0.1)
    # echo_intensity_map = speckle_reducing_bilateral_filter(n_rows, n_colums, echo_intensity_map, 0.05)

    return echo_intensity_map, inverse_intensity_map
    #return echo_intensity_map, probability_map
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
    
    # n_dim = max(n_rows,n_colums)
    # padded_intensity_mean = fill(NaN, n_dim,n_dim)

    # temp = copy(intensity_mean)
    # padded_intensity_mean[1:n_rows,1:n_colums] = temp
    # mask = isnan.(padded_intensity_mean)
    # replace!(padded_intensity_mean, NaN=>0.0)

    # filtered_image = anisotropic_diffusion(padded_intensity_mean, kappa=20, gamma=0.25, option=1)
    # filtered_image[mask] .= NaN;

    # return (intensity_mean, intensity_variance, filtered_image)
    return (intensity_mean, intensity_variance, intensity_mean)
end

function anisotropic_diffusion(img; niter=1, kappa=50, gamma=0.1, voxelspacing=nothing, option=1)
    # define conduction gradients functions
    if option == 1
        condgradient(delta, spacing) = exp(-(delta/kappa)^2.)/spacing
    #elseif option == 2
    #    condgradient(delta, spacing) = 1.0/(1.0+(delta/kappa)^2.0)/Float64(spacing)
    #elseif option == 3
    #    kappa_s = kappa * (2**0.5)
    #    condgradient(delta, spacing) = ifelse(abs.(delta) .<= kappa_s, 0.5*((1.-(delta/kappa_s)**2.)**2.)/Float64(spacing), 0.0)
    end
    # initialize output array
    out = img

    # set default voxel spacing if not supplied
    if voxelspacing == nothing
        voxelspacing = ones(Float64, length(size(out)))
    end
    # initialize some internal variables
    deltas = [zeros(Float64,size(out)) for k ∈ 1:length(size(out))]

    for _k ∈ 1:niter

        # calculate the diffs
        for i ∈ 1:length(size(out))
            slicer = []
            for j ∈ 1:length(size(out))
                append!(slicer,j==i ? [[1:size(out)[j]-1...]] : [[1:size(out)[j]...]])
            end
            deltas[i][slicer...] = diff(out,dims=i)
        end
        # update matrices
        matrices = [condgradient(deltas[i], voxelspacing[i]) * deltas[i] for i ∈ 1:length(deltas)]
        # subtract a copy that has been shifted ('Up/North/West' in 3D case) by one
        # pixel. Don't as questions. just do it. trust me.
        for i ∈ 1:length(size(out))
            slicer = []
            for j ∈ 1:length(size(out))
                append!(slicer,j==i ? [[2:size(out)[j]...]] : [[1:size(out)[j]...]])
            end
            matrices[i][slicer...] = diff(matrices[i],dims=i)
        end
        # update the image
        out = out + gamma * sum(matrices)
    end
    return out
end


function speckle_reducing_bilateral_filter(n_rows, n_colums, map, standard_deviation)

    filtered_map = fill(NaN, (n_rows, n_colums))

    spatial_kernel = ImageFiltering.Kernel.gaussian(standard_deviation)
    kernel_size = size(spatial_kernel)[1]

    padded_map = Images.padarray(map, Images.Pad(:reflect,Int((floor(kernel_size/2))), Int(floor(kernel_size/2))))

    for row=1:n_rows, col=1:n_colums
        if isnan(map[row, col])
            continue
        end

        current_cell_int = map[row, col]

        normalization_factor = 0.0
        filtered_cell = 0.0

        for i=-Int((floor(kernel_size/2))):Int((floor(kernel_size/2))), j=-Int((floor(kernel_size/2))):Int((floor(kernel_size/2)))
            if isnan(padded_map[row+i,col+j])
                continue
            end

            alpha_squared = (padded_map[row+i,col+j] ^ 2) / 2
            range_support = (current_cell_int/alpha_squared) * exp(-(current_cell_int^2)/(2*alpha_squared))
            filtered_cell += padded_map[row+i,col+j] * spatial_kernel[i,j] * range_support
            normalization_factor += spatial_kernel[i,j] * range_support
        end

        filtered_map[row, col] = filtered_cell / normalization_factor

    end  

    return filtered_map

end

end
