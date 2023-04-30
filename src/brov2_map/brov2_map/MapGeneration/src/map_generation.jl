module MapGenerationFunctions

using StaticArrays
using LinearAlgebra
using TimerOutputs
using DataStructures

import NearestNeighbors
import Distances
import Statistics
import Distributions
import Images
import ImageFiltering


mutable struct Swath
    data_port::Vector{Float64}
    data_stb::Vector{Float64}
    odom::SVector{5,Float64} #(x,y,roll,pitch,yaw)
    altitude::Float64
end

always_false(i::Int) = false
const four_nn = [SVector{2,Int}(-1, 0), SVector{2,Int}(1, 0), SVector{2,Int}(0, -1), SVector{2,Int}(0, 1)]

function generate_map(n_rows, n_colums, n_bins, map_resolution, map_origin_x, map_origin_y, 
                      swaths, sonar_range, swath_ground_resolution, probability_threshold)
    
    to = TimerOutput()

    swath_slant_resolution = 0.03
    sonar_theta = (25 * pi) / 180
    sonar_alpha = pi / 3
    sonar_beta = (0.5 * pi) / 180
    sonar_x_offset = -0.2532
    sonar_y_offset = 0.082
    sonar_z_offset = 0.033
        
    map_origin = SVector(map_origin_x, map_origin_y)        
    swath_locals = [Swath(
        swath.data_port, 
        swath.data_stb, 
        SVector(
            swath.odom[1] + (sonar_x_offset * cos(swath.odom[4])
                        + (swath.altitude - sonar_z_offset 
                        + sonar_x_offset * sin(swath.odom[4]))
                        * sin(swath.odom[4])) 
                        * cos(swath.odom[5]), # Pitch correction
            swath.odom[2] + (sonar_x_offset * cos(swath.odom[4])
                        + (swath.altitude - sonar_z_offset
                        + sonar_x_offset * sin(swath.odom[4])) 
                        * sin(swath.odom[4])) 
                        * sin(swath.odom[5]), # Pitch correction
            swath.odom[3],
            swath.odom[4],
            swath.odom[5]
        ),
        float(swath.altitude)) 
        for swath in swaths
    ]

    intensity_map = fill(NaN, (n_rows, n_colums))

    for _ in 1:3

        # Optimized method
        knn_k = 2
        knn_max_dist = 0.2
        knn_max_variance = 0.05

        intensity_map = fill(NaN, (n_rows, n_colums))

        probabilities = Array{Vector{Float64}}(undef, n_rows, n_colums)
        intensities = Array{Vector{Float64}}(undef, n_rows, n_colums)
        indexes = zeros(Int, n_rows, n_colums)

        buffer_size = Int(ceil(length(swath_locals)))
        # buffer_size = Int(ceil(length(swath_locals) * 0.35))
        # buffer_size = 400

        for i=1:n_rows, j=1:n_colums
            probabilities[i,j] = zeros(Float64, buffer_size)
            intensities[i,j] = zeros(Float64, buffer_size)
        end

        cell_coordinates = fill(SVector(0.0,0.0), n_rows*n_colums)
        intensity_values = zeros(n_rows*n_colums)

        cell_transformations = Array{SVector{2,Float64}}(undef, n_rows+1, n_colums+1)
        cell_visited = Array{Bool}(undef, n_rows, n_colums)
        cells_to_filter = fill(false, (n_rows, n_colums))
        cells_to_visit = CircularDeque{SVector{2,Int}}(Int((20*2*sonar_range)/map_resolution))

        @timeit to "map_generation_opt" generate_map_optimized!(
            n_rows, n_colums, n_bins, map_resolution, 
            map_origin, swath_locals, sonar_range, probability_threshold,
            sonar_beta, swath_slant_resolution,
            knn_k, knn_max_dist, knn_max_variance,
            intensity_map, probabilities, intensities, cell_transformations,  
            cell_visited, cells_to_visit, indexes, cell_coordinates, intensity_values, cells_to_filter, 
            sonar_theta, sonar_alpha, sonar_x_offset, sonar_y_offset, sonar_z_offset)


        # Original method
        knn_k = 2
        knn_max_dist = 0.2
        knn_max_variance = 0.05

        intensity_map = fill(NaN, (n_rows, n_colums))
        buffer_size = length(swath_locals)
        probabilities = zeros(Float64, buffer_size)
        intensities = zeros(Float64, buffer_size) 

        cell_coordinates = fill(SVector(0.0,0.0), n_rows*n_colums)
        intensity_values = zeros(n_rows*n_colums)
        cells_to_filter = fill(false, (n_rows, n_colums))

        cell_local = Array{SVector{2,Float64}}(undef, 2, 2)

        @timeit to "map_generation_org" generate_map_original!(
            n_rows, n_colums, n_bins, map_resolution, 
            map_origin, swath_locals, sonar_range, probability_threshold,
            sonar_beta, swath_ground_resolution,
            knn_k, knn_max_dist, knn_max_variance,
            intensity_map, cell_local, 
            probabilities, intensities, cells_to_filter,
            cell_coordinates, intensity_values
        )
        
        # Raw knn
        knn_k = 4
        knn_max_dist = 0.3
        knn_max_variance = 0.05

        intensity_map = fill(NaN, (n_rows, n_colums))

        bin_coordinates = fill(SVector(0.0,0.0), 2 * n_bins * length(swath_locals))
        intensity_values = zeros(2 * n_bins * length(swath_locals))

        @timeit to "map_generation_knn" generate_map_knn!(
            n_rows, n_colums, n_bins, map_resolution, 
            map_origin, swath_ground_resolution, knn_k, knn_max_dist, knn_max_variance,
            swath_locals, intensity_map, bin_coordinates, intensity_values)

    end

    show(to)

    # intensity_map = speckle_reducing_bilateral_filter(intensity_map, 0.1)
    # intensity_map = speckle_reducing_bilateral_filter(intensity_map, 0.3)
    # intensity_map = speckle_reducing_bilateral_filter(intensity_map, 0.5)

    # echo_intensity_map = bilateral_filter(echo_intensity_map, 0.1, 0.5)
    # echo_intensity_map = bilateral_filter(echo_intensity_map, 0.3, 0.7)
    # echo_intensity_map = bilateral_filter(echo_intensity_map, 0.5, 0.9)

    return intensity_map
end
 

function generate_map_optimized!(n_rows, n_colums, n_bins, map_resolution, 
    map_origin, swath_locals, sonar_range, probability_threshold,
    sonar_beta, swath_slant_resolution,
    knn_k, knn_max_dist, knn_max_variance,
    intensity_map, probabilities, intensities, cell_transformations,  
    cell_visited, cells_to_visit, indexes, cell_coordinates, intensity_values, cells_to_filter, 
    sonar_theta, sonar_alpha, sonar_x_offset, sonar_y_offset, sonar_z_offset)

    for swath in swath_locals

        # Array containing the polar coordinates of the cell corners relative the the current measurement
        cell_transformations = fill!(cell_transformations, SVector(NaN, NaN)) 
        cell_visited = fill!(cell_visited, false) 

        # Precompute some values
        cos_roll = cos(swath.odom[3])
        sin_roll = sin(swath.odom[3])
        cos_pitch = cos(swath.odom[4])
        sin_pitch = sin(swath.odom[4])
        angle_fbr = sonar_theta + sonar_alpha / 2

        # # For non corrected altitude
        # corr_altitude_port = ((swath.altitude - sonar_z_offset) * 
        #                        cos_roll * cos_pitch) + 
        #                        sonar_x_offset * sin_pitch +
        #                        sonar_y_offset * sin_roll
        # corr_altitude_stb = ((swath.altitude - sonar_z_offset) * 
        #                       cos_roll * cos_pitch) + 
        #                       sonar_x_offset * sin_pitch -
        #                       sonar_y_offset * sin_roll

        # For attitude corrected altitude
        corr_altitude_port = swath.altitude -  
                                sonar_z_offset * cos_roll * cos_pitch + 
                                sonar_x_offset * sin_pitch +
                                sonar_y_offset * sin_roll
        corr_altitude_stb = swath.altitude - 
                            sonar_z_offset * cos_roll * cos_pitch + 
                            sonar_x_offset * sin_pitch -
                            sonar_y_offset * sin_roll

        fbr_slant_range_port = corr_altitude_port / 
                                sin(angle_fbr - swath.odom[3])
        fbr_slant_range_stb = corr_altitude_stb / 
                                sin(angle_fbr + swath.odom[3])

        fbr_ground_range_port = sqrt(fbr_slant_range_port ^ 2 - corr_altitude_port ^ 2) 
        fbr_ground_range_stb = sqrt(fbr_slant_range_stb ^ 2 - corr_altitude_stb ^ 2)

        sonar_ground_range_port = sqrt(sonar_range ^ 2 - corr_altitude_port ^ 2) 
        sonar_ground_range_stb = sqrt(sonar_range ^ 2 - corr_altitude_stb ^ 2)

        horisontal_y_offset = sonar_y_offset * cos_roll +
                                sonar_z_offset * sin_roll

        empty!(cells_to_visit)

        # Find first cell outside blindzone on port and stb 
        row_col_port = SVector(
            Int(ceil(
                -(swath.odom[1] - map_origin[1] + 
                fbr_ground_range_port * cos(swath.odom[5] - pi/2)) / 
                map_resolution
            )),
            Int(ceil(
                (swath.odom[2] - map_origin[2] + 
                fbr_ground_range_port * sin(swath.odom[5] - pi/2)) / 
                map_resolution
            ))
        )

        row_col_stb = SVector(
            Int(ceil(
                -(swath.odom[1] - map_origin[1] + 
                fbr_ground_range_stb * cos(swath.odom[5] + pi/2)) / 
                map_resolution
            )),
            Int(ceil(
                (swath.odom[2] - map_origin[2] + 
                fbr_ground_range_stb * sin(swath.odom[5] + pi/2)) / 
                map_resolution
            ))
        )

        # Add cell with and do 8 connectivity
        for i in -1:1, j in -1:1
            push!(cells_to_visit, row_col_port + SVector(i,j))
            push!(cells_to_visit, row_col_stb + SVector(i,j))
        end

        while !isempty(cells_to_visit)
            row, colum = pop!(cells_to_visit)

            if cell_visited[row ,colum]
                continue
            end

            cell_visited[row ,colum] = true

            calculate_cell_measurement_transformation!(
                cell_transformations, row, colum, swath.odom, map_origin, map_resolution
            )

            prob_observation = get_cell_probability_gaussian(
                cell_transformations, row, colum, sonar_beta, 
                sonar_ground_range_port, sonar_ground_range_stb,
                fbr_ground_range_port, fbr_ground_range_stb
            ) :: Float64

            if prob_observation >= probability_threshold
                intensity = get_cell_intensity_non_corr_swath(
                    cell_transformations, row, colum, 
                    swath, swath_slant_resolution, n_bins, horisontal_y_offset,
                    corr_altitude_port, corr_altitude_stb,
                    fbr_slant_range_port, fbr_slant_range_stb
                ) :: Float64

                if !isnan(intensity)
                    cells_to_filter[row,colum] = false
                    indexes[row, colum] += 1
                    probabilities[row, colum][indexes[row,colum]] = prob_observation
                    intensities[row, colum][indexes[row,colum]] = intensity
                end

                for i=1:4
                    new_cell = SVector{2,Int}(row,colum) + four_nn[i]
                    if checkbounds(Bool, cell_visited, new_cell[1], new_cell[2])
                        push!(cells_to_visit, new_cell)
                    end
                end
                
            elseif iszero(indexes[row,colum]) && prob_observation > 0.0 
                cells_to_filter[row,colum] = true
            end 
        end
    end

    for row=1:n_rows, colum=1:n_colums

        if indexes[row,colum] == 0
            continue
        end
        
        intensity_map[row, colum] = dot(
            view(intensities[row, colum], 1:indexes[row,colum]),
            view(probabilities[row, colum], 1:indexes[row,colum]) / 
            sum(view(probabilities[row, colum], 1:indexes[row,colum]))
        )
    end

    knn_filtering!(
        n_rows, n_colums, map_resolution, knn_k, knn_max_dist, knn_max_variance,
        intensity_map, cell_coordinates, intensity_values, cells_to_filter
    )
end


function calculate_cell_measurement_transformation!(cell_transformations, row, colum, 
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
            cell_transformations[row+i, colum+j] = SVector(
                norm(v), # r
                rem2pi((atan(v[2], v[1]) - measurement_frame[5]), RoundDown) # theta
            )
        end
    end
end


function get_cell_probability_gaussian(
    cell_transformations, row, colum, sonar_beta, 
    sonar_ground_range_port, sonar_ground_range_stb,
    fbr_ground_range_port, fbr_ground_range_stb)

    # Correct differently for port and starboard swath 
    # Transducers assumed to be pointing in yz-plane of body frame
    is_port = cell_transformations[row,colum][2] > pi
    correction = pi/2 + is_port * pi

    max_r = 0.0
    min_theta = 100.0*pi
    max_theta = -100.0*pi

    for i=0:1, j=0:1
        max_r = max(max_r, cell_transformations[row+i, colum+j][1])
        new_theta = cell_transformations[row+i, colum+j][2] - correction
        min_theta = min(min_theta, new_theta)
        max_theta = max(max_theta, new_theta)
    end
 
    if max_r < (is_port ? sonar_ground_range_port : sonar_ground_range_stb) && 
       max_r > (is_port ? fbr_ground_range_port : fbr_ground_range_stb) &&
       max_theta - min_theta < pi 
        d = Distributions.Normal(0.0, sonar_beta)
        return Distributions.cdf(d, max_theta) - Distributions.cdf(d, min_theta)
    else
        return 0.0
    end
end


function get_cell_intensity_non_corr_swath(cell_transformations, row, colum, 
    swath, swath_slant_resolution, n_bins, horisontal_y_offset,
    corr_altitude_port, corr_altitude_stb,
    fbr_slant_range_port, fbr_slant_range_stb)

    pixel_intensity = 0.0
    valid_corners = 0

    corrected_altitude = (cell_transformations[row, colum][2] < pi) ?
                         corr_altitude_stb :
                         corr_altitude_port

    first_bottom_return_range = (cell_transformations[row, colum][2] < pi) ?
                                fbr_slant_range_stb : 
                                fbr_slant_range_port

    for i=0:1, j=0:1
        slant_range = sqrt(
            (cell_transformations[row+i, colum+j][1] - horisontal_y_offset) ^ 2 + 
            corrected_altitude ^ 2
        )

        if slant_range < first_bottom_return_range
            continue
        end

        # Pre-compute index and weights
        index = slant_range / swath_slant_resolution
        lower_index = Int(floor(index))
        higher_index = lower_index + 1 # For optimization
        w1 = index - lower_index
        w2 = 1.0 - w1

        # The corner is outside the swath range or not measured
        if higher_index > n_bins || lower_index == 0
            continue
        end

        # Interpolate
        measure_intensity = (cell_transformations[row+i, colum+j][2] < pi) ?
            (w1 * swath.data_stb[higher_index] + 
            w2 * swath.data_stb[lower_index]) :
            (w1 * swath.data_port[n_bins - higher_index + 1] + 
            w2 * swath.data_port[n_bins - lower_index + 1])

        # Do not use corner if it evaluates to NaN
        if isnan(measure_intensity)
            continue
        end

        pixel_intensity += measure_intensity
        valid_corners += 1
    end

    return pixel_intensity / valid_corners
end


function generate_map_original!(
    n_rows, n_colums, n_bins, map_resolution, 
    map_origin, swath_locals, sonar_range, probability_threshold,
    sonar_beta, swath_ground_resolution,
    knn_k, knn_max_dist, knn_max_variance,
    intensity_map, cell_local, 
    probabilities, intensities, cells_to_filter,
    cell_coordinates, intensity_values
)

    cell_global = SVector{2, Float64}

    for row=1:n_rows, col=1:n_colums

        cell_global = SVector(
            map_origin[1] - (row - 1) * map_resolution,
            map_origin[2] + (col - 1) * map_resolution
        ) 
        prob_observation = NaN
        index = 0

        for swath in swath_locals

            get_cell_coordinates!(
                cell_local, cell_global, swath.odom, map_resolution
            )
            prob_observation = get_cell_probability_gaussian(
                cell_local, sonar_range, sonar_beta
            ) :: Float64

            if prob_observation >= probability_threshold
                cells_to_filter[row,col] = false

                intensity = get_cell_echo_intensity(
                    cell_local, swath, swath_ground_resolution, n_bins
                ) :: Float64

                if !isnan(intensity)
                    index += 1
                    intensities[index] = intensity
                    probabilities[index] = prob_observation
                end
            elseif iszero(index) && 
                    prob_observation > probability_threshold * 0.5
                cells_to_filter[row,col] = true
            end
        end

        if index > 0
            intensity_map[row, col] = dot(
                view(intensities, 1:index),
                view(probabilities, 1:index) / sum(view(probabilities, 1:index))
            )
        end
    end

    knn_filtering!(
            n_rows, n_colums, map_resolution, knn_k, knn_max_dist, knn_max_variance,
            intensity_map, cell_coordinates, intensity_values, cells_to_filter
        )
end


function get_cell_coordinates!(cell_local, cell_global, measurement_frame, map_resolution)

    cell_measurement_transformation = SVector(
        cell_global[1] - measurement_frame[1],
        cell_global[2] - measurement_frame[2]
    )

    # Transform the transformation to polar coordinates centered in the measurement frame
    for i=1:2, j=1:2
        v = cell_measurement_transformation + SVector(-(i-1) * map_resolution, (j-1) * map_resolution)
        cell_local[i, j] = SVector(
            norm(v), # r
            rem2pi((atan(v[2], v[1]) - measurement_frame[5]), RoundDown) # theta
        )
    end
end


function get_cell_probability_gaussian(cell_local, sonar_range, sonar_beta)
    
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
        d = Distributions.Normal(0.0, sonar_beta)
        return Distributions.cdf(d, max_theta) - Distributions.cdf(d, min_theta)
    else
        return 0.0
    end
end
    

function get_cell_echo_intensity(cell_local, swath, swath_resolution, n_bins)
    
    pixel_intensity = 0.0
    valid_corners = 0

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
        measure_intensity = (cell_local[i, j][2] < pi) ?
            (w1 * swath.data_stb[higher_index] + w2 * swath.data_stb[lower_index]) :
            (w1 * swath.data_port[n_bins-higher_index+1] + w2 * swath.data_port[n_bins-lower_index+1])
            
        # Do not use corner if it evaluates to NaN
        if isnan(measure_intensity)
            continue
        end

        pixel_intensity += measure_intensity
        valid_corners += 1
    end

    return pixel_intensity / valid_corners
end

function generate_map_knn!(
    n_rows, n_colums, n_bins, map_resolution, 
    map_origin ::SVector{2,Float64}, swath_ground_resolution, knn_k, knn_max_dist, knn_max_variance,
    swath_locals, intensity_map, bin_coordinates, intensity_values)

    n_data_points = 0

    for swath in swath_locals

        bin_increment_port = SVector(
            swath_ground_resolution * cos(swath.odom[5] - pi/2),
            swath_ground_resolution * sin(swath.odom[5] - pi/2)
        )
        bin_increment_stb = SVector(
            swath_ground_resolution * cos(swath.odom[5] + pi/2),
            swath_ground_resolution * sin(swath.odom[5] + pi/2)
        )
        bin_coordinate_port = SVector(swath.odom[1], swath.odom[2])
        bin_coordinate_stb = bin_coordinate_port

        for bin=1:n_bins

            if !isnan(swath.data_port[n_bins - bin + 1])
                n_data_points += 1
                bin_coordinates[n_data_points] = bin_coordinate_port
                intensity_values[n_data_points] = swath.data_port[n_bins - bin + 1]
            end 
            bin_coordinate_port += bin_increment_port

            if !isnan(swath.data_stb[bin])
                n_data_points += 1
                bin_coordinates[n_data_points] = bin_coordinate_stb
                intensity_values[n_data_points] = swath.data_stb[bin]
            end 
            bin_coordinate_stb += bin_increment_stb
            
        end
    end

    kdtree = NearestNeighbors.KDTree(
        view(bin_coordinates, 1:n_data_points), 
        Distances.Euclidean()
    )

    idx = Vector{Int}(undef, knn_k)
    dist = Vector{Float64}(undef, knn_k)
    svals = SVector{knn_k, Float64}

    for row=1:n_rows, col=1:n_colums

        NearestNeighbors.knn_point!(
            kdtree,
            map_origin + SVector{2,Float64}(-(row-1)*map_resolution, (col-1)*map_resolution),
            false, dist, idx, always_false
        )

        svals = view(intensity_values, idx[dist .<= knn_max_dist])

        if length(svals) > 0
            var = Statistics.var(svals)
            if var <= knn_max_variance
                intensity_map[row,col] = Statistics.mean(svals)
            else
                intensity_map[row,col] = Statistics.quantile!(svals, 10/100)
            end
        end
    end
end

function knn_filtering!(n_rows, n_colums, map_resolution, knn_k, knn_max_dist, knn_max_variance,
    intensity_map, cell_coordinates, intensity_values, cells_to_filter)
    
    n_cells = 0

    for row=1:n_rows, col=1:n_colums
        if isnan(intensity_map[row,col])
            continue
        end

        n_cells += 1
        cell_coordinates[n_cells] = SVector((row-1)*map_resolution, (col-1)*map_resolution)
        intensity_values[n_cells] = intensity_map[row,col]
    end

    kdtree = NearestNeighbors.KDTree(
        view(cell_coordinates, 1:n_cells), 
        Distances.Euclidean()
    )
    
    idx = Vector{Int}(undef, knn_k)
    dist = Vector{Float64}(undef, knn_k)
    svals = SVector{knn_k, Float64}

    for row=1:n_rows, col=1:n_colums
        if !cells_to_filter[row,col] || !isnan(intensity_map[row,col])
            continue
        end

        NearestNeighbors.knn_point!(
            kdtree,
            SVector((row-1)*map_resolution, (col-1)*map_resolution),
            false, dist, idx, always_false
        )

        svals = view(intensity_values, idx[dist .<= knn_max_dist])

        if length(svals) > 0
            var = Statistics.var(svals)
            if var <= knn_max_variance
                intensity_map[row,col] = Statistics.mean(svals)
            else
                intensity_map[row,col] = Statistics.quantile(svals, 10/100)
            end
        end
    end
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