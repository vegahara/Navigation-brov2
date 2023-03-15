module MapGenerationFunctions

using StaticArrays
using LinearAlgebra
using TimerOutputs

import NearestNeighbors
import Distances
import Statistics
import Distributions
import Images
import ImageFiltering


mutable struct Swath
    data_port
    data_stb
    odom::SVector{5,Float64} #(x,y,roll,pitch,yaw)
    altitude::Float64
end


function generate_map(n_rows, n_colums, n_bins, map_resolution, map_origin_x, map_origin_y, 
    swaths, sonar_range, sonar_horizontal_beam_spread, swath_ground_resolution)
    
    to = TimerOutput()
        
    map_origin = SVector(map_origin_x, map_origin_y)        
    swath_locals = [Swath(
        swath.data_port, 
        swath.data_stb, 
        SVector(
            swath.odom[1]+swath.altitude*sin(swath.odom[4])*cos(swath.odom[5]), # Pitch correction
            swath.odom[2]+swath.altitude*sin(swath.odom[4])*sin(swath.odom[5]), # Pitch correction
            swath.odom[3],
            swath.odom[4],
            swath.odom[5]
        ),
        float(swath.altitude)) 
        for swath in swaths
    ]

    probability_map = ones(n_rows, n_colums)
    intensity_map = fill(NaN, (n_rows, n_colums))
    range_map = fill(NaN, (n_rows, n_colums))
    altitude_map = fill(NaN, (n_rows, n_colums))
    observed_swaths = fill(Int[], (n_rows, n_colums))

    # Optimized method
    probabilities = fill(Float64[], (n_rows, n_colums))
    intensities = fill(Float64[], (n_rows, n_colums))
    altitudes = fill(Float64[], (n_rows, n_colums))
    ranges = fill(Float64[], (n_rows, n_colums))  
    cell_transformations = Array{SVector{2,Float64}}(undef, n_rows+1, n_colums+1)
    cell_visited = Array{Bool}(undef, n_rows, n_colums)

    intensity_map, probability_map, observed_swaths, range_map = @timeit to "map_generation_opt" generate_map_optimized(
        n_rows, n_colums, n_bins, map_resolution, 
        map_origin, swath_locals, sonar_range, 
        sonar_horizontal_beam_spread, swath_ground_resolution,
        probability_map, intensity_map, range_map, altitude_map,
        observed_swaths, probabilities, intensities, altitudes, ranges,
        cell_transformations, cell_visited, to
    )

    # # Original method
    # cell_local = Array{MVector{2,Float64}}(undef, 2, 2)

    # intensity_map, probability_map, observed_swaths, range_map = @timeit to "map_generation_org" generate_map_original(
    #     n_rows, n_colums, n_bins, map_resolution, 
    #     map_origin, swath_locals, sonar_range, 
    #     sonar_horizontal_beam_spread, swath_ground_resolution,
    #     probability_map, intensity_map, range_map, altitude_map,
    #     observed_swaths, cell_local, to
    # )

    show(to)

    # intensity_map = speckle_reducing_bilateral_filter(intensity_map, 0.1)
    # intensity_map = speckle_reducing_bilateral_filter(intensity_map, 0.3)
    # intensity_map = speckle_reducing_bilateral_filter(intensity_map, 0.5)

    # echo_intensity_map = bilateral_filter(echo_intensity_map, 0.1, 0.5)
    # echo_intensity_map = bilateral_filter(echo_intensity_map, 0.3, 0.7)
    # echo_intensity_map = bilateral_filter(echo_intensity_map, 0.5, 0.9)

    return intensity_map, probability_map, observed_swaths, range_map
end
 

function generate_map_optimized(n_rows, n_colums, n_bins, map_resolution, 
    map_origin, swath_locals, sonar_range, 
    sonar_horizontal_beam_spread, swath_ground_resolution,
    probability_map, intensity_map, range_map, altitude_map,
    observed_swaths, probabilities, intensities, altitudes, ranges,
    cell_transformations, cell_visited, to)

    @timeit to "swath_iteration" begin

        for (swath, swath_index) in zip(swath_locals, 1:length(swath_locals))

            @timeit to "swath_allocation" begin

                # Array containing the polar coordinates of the cell corners relative the the current measurement
                cell_transformations = fill!(cell_transformations, SVector(NaN, NaN)) 
                cell_visited = fill!(cell_visited, false) 

                @views data_port = reverse(swath.data_port)
                @views data_stb = swath.data_stb

                # Find cell map coordinates of the sonar base
                v = swath.odom[1:2] - map_origin
                row = Int(ceil(-v[1] / map_resolution))
                colum = Int(ceil(v[2] / map_resolution))

                cells_to_visit = []

                # Do 25 connectivity for first cell (8 is not sufficient)
                for i in -2:2, j in -2:2
                    push!(cells_to_visit, [row+i, colum+j])
                end  
            end

            @timeit to "cell_iteration" begin

                while !isempty(cells_to_visit)
                    row, colum = popfirst!(cells_to_visit)
                    cell_visited[row ,colum] = true

                    @timeit to "cell_meas_trans" calculate_cell_measurement_transformation(
                        cell_transformations, row, colum, swath.odom, map_origin, map_resolution
                    )

                    prob_observation = @timeit to "cell_probability" get_cell_probability_gaussian(
                        cell_transformations, row, colum, sonar_range, sonar_horizontal_beam_spread
                    ) :: Float64

                    if prob_observation >= 0.1
                        intensity = @timeit to "cell_intensity" get_cell_intensity(
                            cell_transformations, row, colum, data_port, data_stb, 
                            swath_ground_resolution, n_bins
                        ) :: Float64

                        if !isnan(intensity)
                            observed_swaths[row, colum] = [observed_swaths[row, colum];[swath_index - 1]] # 0 indexed
                            probabilities[row, colum] = [probabilities[row, colum];[prob_observation]]
                            intensities[row, colum] = [intensities[row, colum];[intensity]]
                            altitudes[row, colum] = [altitudes[row, colum];[swath.altitude]]
                            ranges[row, colum] = [
                                ranges[row, colum];
                                [Statistics.mean(cell_transformations[row:row+1, colum:colum+1][1])]
                            ]
                        end

                        # Add 4-connected neighbor cells to cells_to_visit
                        for new_cell in [[row-1, colum],[row+1, colum],[row, colum-1],[row, colum+1]]
                            if checkbounds(Bool, cell_visited, new_cell[1], new_cell[2]) &&
                            !cell_visited[new_cell[1], new_cell[2]] &&
                            !(new_cell in cells_to_visit)
                                push!(cells_to_visit, new_cell)
                            end
                        end 
                    end    
                end
            end
        end
    end

    @timeit to "map_iteration" begin
        for row=1:n_rows, colum=1:n_colums

            if isempty(probabilities[row, colum])
                continue
            end

            intensity_map[row, colum] = dot(
                intensities[row, colum],
                probabilities[row, colum] / sum(probabilities[row, colum])
            )

            probability_map[row, colum] *= prod(1 .- probabilities[row, colum])

            range_map[row, colum] = Statistics.mean(ranges[row, colum])
            
            altitude_map[row, colum] = Statistics.mean(altitudes[row, colum])
        end
    end

    @timeit to "map_interpolation" begin

        k = 4;
        variance_ceiling = 0.05
        max_distance = 0.2

        intensity_map, intensity_variance = knn(
            n_rows, n_colums, intensity_map, 
            map_resolution, k, variance_ceiling, max_distance
        )

    end
            
    return intensity_map, probability_map, observed_swaths, range_map
end


function calculate_cell_measurement_transformation(cell_transformations, row, colum, 
                                                   measurement_frame, map_origin, map_resolution)

    # Get the x,y transformation from the measurement to the map cell in global coordinates
    cell_measurement_transformation = SVector(
        (map_origin[1] - (row - 1) * map_resolution - measurement_frame[1]),
        (map_origin[2] + (colum - 1) * map_resolution - measurement_frame[2]) 
    )

    # Transform the transformation to polar coordinates centered in the measurement frame
    for i=0:1, j=0:1
        if isnan(cell_transformations[row+i, colum+j][1])
            v = cell_measurement_transformation + SVector(-i * map_resolution, j * map_resolution)
            r = norm(v)
            theta = rem2pi((atan(v[2], v[1]) - measurement_frame[5]), RoundDown)
            cell_transformations[row+i, colum+j] = SVector(r, theta)
        end
    end
end


function get_cell_probability_gaussian(cell_transformations, row, colum, 
                                       sonar_range, sonar_horizontal_beam_spread)
    
    # Correct differently for port and starboard swath 
    # Transducers assumed to be pointing in yz-plane of body frame
    correction = pi/2 + (cell_transformations[row,colum][2] > pi)*pi

    max_r = 0.0
    min_theta = 100.0*pi
    max_theta = -100.0*pi

    for i=0:1, j=0:1
        max_r = max(max_r, cell_transformations[row+i, colum+j][1])
        new_theta = cell_transformations[row+i, colum+j][2] - correction
        min_theta = min(min_theta, new_theta)
        max_theta = max(max_theta, new_theta)
    end

    if max_r < sonar_range && (max_theta - min_theta) < pi
        d = Distributions.Normal(0.0, sonar_horizontal_beam_spread)
        return Distributions.cdf(d, max_theta) - Distributions.cdf(d, min_theta)
    else
        return 0.0
    end
end


function get_cell_intensity(cell_transformations, row, colum, 
                            data_port, data_stb, swath_resolution, n_bins)

    pixel_intensity = 0.0
    valid_corners = 0

    for i=0:1, j=0:1
        # Pre-compute index and weights
        index = cell_transformations[row+i, colum+j][1]/swath_resolution
        lower_index = Int(floor(index))
        higher_index = lower_index + 1 # For optimization
        w1 = index - lower_index
        w2 = 1.0 - w1

        # The corner is outside the swath range or not measured
        if higher_index > n_bins || lower_index == 0
            continue
        end

        # Interpolate
        measure_intensity = (cell_transformations[row+i, colum+j][2] > pi) ?
                            (w1 * data_stb[higher_index] + w2 * data_stb[lower_index]) :
                            (w1 * data_port[higher_index] + w2 * data_port[lower_index])

        # Do not use corner if it evaluates to NaN
        if isnan(measure_intensity)
            continue
        end

        pixel_intensity += measure_intensity
        valid_corners += 1
    end

    return pixel_intensity / valid_corners
end


function generate_map_original(n_rows, n_colums, n_bins, map_resolution, 
    map_origin, swath_locals, sonar_range, 
    sonar_horizontal_beam_spread, swath_ground_resolution,
    probability_map, intensity_map, range_map, altitude_map,
    observed_swaths, cell_local, to)

    @timeit to "cell_iteration" begin
        for row=1:n_rows, col=1:n_colums
            @timeit to "cell_alloc" begin
                cell_global = SVector(
                    map_origin[1] - (row - 1) * map_resolution,
                    map_origin[2] + (col - 1) * map_resolution
                ) 
                echo_intensities = []
                swath_probabilities = []
                altitudes = []
                observed_swaths_cell = []
                cell_ranges = []
                probability = 1.0
                cell_observed = false 
                prob_observation = NaN
            end

            @timeit to "swath_iteration" begin

                for (swath, swath_number) in zip(swath_locals, 1:length(swath_locals))

                    @timeit to "cell_meas_trans" get_cell_coordinates(
                        cell_local, cell_global, swath.odom, map_resolution
                    )
                    prob_observation = @timeit to "cell_probability" get_cell_probability_gaussian(
                        cell_local, sonar_range, sonar_horizontal_beam_spread
                    ) :: Float64

                    if prob_observation >= 0.1
                        echo_intensity_swath = @timeit to "cell_intensity" get_cell_echo_intensity(
                            cell_local, swath, swath_ground_resolution, n_bins
                        ) :: Float64

                        if !isnan(echo_intensity_swath)
                            cell_observed = true

                            push!(echo_intensities, echo_intensity_swath)
                            push!(swath_probabilities, prob_observation)
                            push!(altitudes, swath.altitude)
                            push!(observed_swaths_cell, swath_number - 1) # 0 indexed
                            push!(cell_ranges, Statistics.mean(cell_local)[1])

                            probability *= (1 - prob_observation)
                        end
                    end
                end
            end

            @timeit to "cell_calculation" begin
                if cell_observed
                    
                    intensity_map[row, col] = dot(
                        echo_intensities,
                        swath_probabilities / sum(swath_probabilities)
                    )
                    probability_map[row,col] = probability
                    range_map[row,col] = Statistics.mean(cell_ranges)
                    altitude_map[row,col] = Statistics.mean(altitudes)
                    observed_swaths[row,col] = observed_swaths_cell
                end
            end
        end
    end

    @timeit to "map_interpolation" begin

        k = 4;
        variance_ceiling = 0.05
        max_distance = 0.2

        intensity_map, intensity_variance = knn(
            n_rows, n_colums, intensity_map, 
            map_resolution, k, variance_ceiling, max_distance
        )

    end
    
    return intensity_map, probability_map, observed_swaths, range_map
end


function get_cell_coordinates(cell_local, cell_global, measurement_frame, map_resolution)

    cell_measurement_transformation = SVector(
        cell_global[1] - measurement_frame[1],
        cell_global[2] - measurement_frame[2]
    )

    # Transform the transformation to polar coordinates centered in the measurement frame
    for i=1:2, j=1:2
        v = cell_measurement_transformation + SVector(-(i-1) * map_resolution, (j-1) * map_resolution)
        r = norm(v)
        theta = rem2pi((atan(v[2], v[1]) - measurement_frame[5]), RoundDown)
        cell_local[i, j] = SVector(r, theta)
    end
end


function get_cell_probability_gaussian(cell_local, sonar_range, sonar_horizontal_beam_spread)
    
    # Correct differently for port and starboard swath 
    # Transducers assumed to be pointing in yz-plane of body frame
    correction = pi/2 + (cell_local[1,1][2] > pi)*pi

    max_r = 0.0
    min_theta = 100.0*pi
    max_theta = -100.0*pi

    for i=1:2, j=1:2
        max_r = max(max_r, cell_local[i,j][1])
        new_theta = cell_local[i,j][2] - correction
        min_theta = min(min_theta, new_theta)
        max_theta = max(max_theta, new_theta)
    end
 
    if max_r < sonar_range && (max_theta - min_theta) < pi
        d = Distributions.Normal(0.0, sonar_horizontal_beam_spread)
        return Distributions.cdf(d, max_theta) - Distributions.cdf(d, min_theta)
    else
        return 0.0
    end
end
    

function get_cell_echo_intensity(cell_local, swath, swath_resolution, n_bins)
    pixel_intensity = 0.0
    valid_corners = 0
    @views data_port = reverse(swath.data_port)
    @views data_stb = swath.data_stb

    for i=1:2, j=1:2
        # Pre-compute index and weights
        index = cell_local[i, j][1]/swath_resolution
        lower_index = Int(floor(index))
        higher_index = lower_index + 1 # For optimization
        w1 = index - lower_index
        w2 = 1.0 - w1

        # The corner is outside the swath range or not measured
        if higher_index > n_bins || lower_index == 0
            continue
        end

        # Interpolate
        measure_intensity = (cell_local[i, j][2] > pi) ?
            (w1 * data_stb[higher_index] + w2 * data_stb[lower_index]) :
            (w1 * data_port[higher_index] + w2 * data_port[lower_index])

        # Do not use corner if it evaluates to NaN
        if isnan(measure_intensity)
            continue
        end

        pixel_intensity += measure_intensity
        valid_corners += 1
    end

    return Statistics.mean(pixel_intensity / valid_corners)
end

function get_cell_intensity(cell_transformations, row, colum, 
    data_port, data_stb, swath_resolution, n_bins)

    pixel_intensity = 0.0
    valid_corners = 0

    for i=0:1, j=0:1
        # Pre-compute index and weights
        index = cell_transformations[row+i, colum+j][1]/swath_resolution
        lower_index = Int(floor(index))
        higher_index = lower_index + 1 # For optimization
        w1 = index - lower_index
        w2 = 1.0 - w1

        # The corner is outside the swath range or not measured
        if higher_index > n_bins || lower_index == 0
        continue
        end

        # Interpolate
        measure_intensity = (cell_transformations[row+i, colum+j][2] > pi) ?
            (w1 * data_stb[higher_index] + w2 * data_stb[lower_index]) :
            (w1 * data_port[higher_index] + w2 * data_port[lower_index])

        # Do not use corner if it evaluates to NaN
        if isnan(measure_intensity)
        continue
        end

        pixel_intensity += measure_intensity
        valid_corners += 1
    end

    return pixel_intensity / valid_corners
end


function knn(n_rows, n_colums, echo_map, map_resolution, k, variance_ceiling, max_distance)
    
    # Make data vectors to use in kdtree
    cell_coordinates = SVector{2,Float64}[]
    intensity_values = []

    for row=1:n_rows, col=1:n_colums
        if !isnan(echo_map[row,col])
            push!(cell_coordinates,SVector{2,Float64}((row-1)*map_resolution, (col-1)*map_resolution))
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
            SVector{2,Float64}((row-1)*map_resolution, (col-1)*map_resolution),
            k
        )
        svals = [intensity_values[idx[ind]] for ind in 1:k if dist[ind] <= max_distance]

        if !isempty(svals)
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