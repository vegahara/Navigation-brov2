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
                      swaths, sonar_range, sonar_horizontal_beam_spread, 
                      swath_ground_resolution, sonar_x_offset, sonar_z_offset)
    
    to = TimerOutput()

    swath_slant_resolution = 0.03
    sonar_theta = (25 * pi) / 180
    sonar_alpha = pi / 3
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


    # Optimized method new
    knn_k = 4
    knn_max_dist = 0.5
    knn_max_variance = 0.05

    # probability_map = ones(n_rows, n_colums)
    probability_map = NaN
    intensity_map = fill(NaN, (n_rows, n_colums))
    # range_map = fill(NaN, (n_rows, n_colums))
    # altitude_map = fill(NaN, (n_rows, n_colums))
    range_map = NaN
    altitude_map = NaN
    # intensity_variance = fill(NaN,n_rows,n_colums)
    intensity_variance = NaN

    # probabilities = fill(Float64[], (n_rows, n_colums))
    # intensities = fill(Float64[], (n_rows, n_colums))
    # altitudes = fill(Float64[], (n_rows, n_colums))
    # ranges = fill(Float64[], (n_rows, n_colums))  

    # observed_swaths = Array{Vector{Int}}(undef, n_rows, n_colums) 
    observed_swaths = NaN
    probabilities = Array{Vector{Float64}}(undef, n_rows, n_colums)
    intensities = Array{Vector{Float64}}(undef, n_rows, n_colums)
    # altitudes = Array{Vector{Float64}}(undef, n_rows, n_colums)
    # ranges = Array{Vector{Float64}}(undef, n_rows, n_colums) 
    altitudes = NaN
    ranges = NaN
    indexes = zeros(Int, n_rows, n_colums)

    # buffer_size = Int(ceil(length(swath_locals)))
    buffer_size = Int(ceil(length(swath_locals) * 0.35))
    # buffer_size = 400

    for i=1:n_rows, j=1:n_colums
        # observed_swaths[i,j] = fill(-1, buffer_size)
        probabilities[i,j] = zeros(Float64, buffer_size)
        intensities[i,j] = zeros(Float64, buffer_size)
        # altitudes[i,j] = zeros(Float64, buffer_size)
        # ranges[i,j] = zeros(Float64, buffer_size)
    end

    cell_coordinates = fill(SVector(0.0,0.0), n_rows*n_colums)
    intensity_values = zeros(n_rows*n_colums)

    println(Int((20*2*sonar_range)/map_resolution))
    println(n_rows)
    println(n_colums)

    cell_transformations = Array{SVector{2,Float64}}(undef, n_rows+1, n_colums+1)
    cell_visited = Array{Bool}(undef, n_rows, n_colums)
    cells_to_visit = CircularDeque{SVector{2,Int}}(Int((20*2*sonar_range)/map_resolution))

    intensity_map, probability_map, observed_swaths, range_map = @timeit to "map_generation_opt_new" generate_map_optimized_new!(
        n_rows, n_colums, n_bins, map_resolution, 
        map_origin, swath_locals, sonar_range, 
        sonar_horizontal_beam_spread, swath_slant_resolution,
        knn_k, knn_max_dist, knn_max_variance,
        probability_map, intensity_map, range_map, altitude_map,
        observed_swaths, probabilities, intensities, altitudes, ranges,
        cell_transformations, intensity_variance, 
        cell_visited, cells_to_visit, indexes, 
        cell_coordinates, intensity_values, 
        sonar_theta, sonar_alpha, sonar_x_offset, sonar_y_offset, sonar_z_offset,to
    )


    # # Optimized method
    # knn_k = 4
    # knn_max_dist = 0.5
    # knn_max_variance = 0.05

    # # probability_map = ones(n_rows, n_colums)
    # probability_map = NaN
    # intensity_map = fill(NaN, (n_rows, n_colums))
    # # range_map = fill(NaN, (n_rows, n_colums))
    # # altitude_map = fill(NaN, (n_rows, n_colums))
    # range_map = NaN
    # altitude_map = NaN
    # # intensity_variance = fill(NaN,n_rows,n_colums)
    # intensity_variance = NaN

    # # probabilities = fill(Float64[], (n_rows, n_colums))
    # # intensities = fill(Float64[], (n_rows, n_colums))
    # # altitudes = fill(Float64[], (n_rows, n_colums))
    # # ranges = fill(Float64[], (n_rows, n_colums))  

    # # observed_swaths = Array{Vector{Int}}(undef, n_rows, n_colums) 
    # observed_swaths = NaN
    # probabilities = Array{Vector{Float64}}(undef, n_rows, n_colums)
    # intensities = Array{Vector{Float64}}(undef, n_rows, n_colums)
    # # altitudes = Array{Vector{Float64}}(undef, n_rows, n_colums)
    # # ranges = Array{Vector{Float64}}(undef, n_rows, n_colums) 
    # altitudes = NaN
    # ranges = NaN
    # indexes = zeros(Int, n_rows, n_colums)

    # buffer_size = Int(ceil(length(swath_locals) * 0.35))

    # for i=1:n_rows, j=1:n_colums
    #     #observed_swaths[i,j] = fill(-1, buffer_size)
    #     probabilities[i,j] = zeros(Float64, buffer_size)
    #     intensities[i,j] = zeros(Float64, buffer_size)
    #     # altitudes[i,j] = zeros(Float64, buffer_size)
    #     # ranges[i,j] = zeros(Float64, buffer_size)
    # end

    # cell_coordinates = fill(SVector(0.0,0.0), n_rows*n_colums)
    # intensity_values = zeros(n_rows*n_colums)

    # cell_transformations = Array{SVector{2,Float64}}(undef, n_rows+1, n_colums+1)
    # cell_visited = Array{Bool}(undef, n_rows, n_colums)
    # cells_to_visit = CircularDeque{SVector{2,Int}}(Int((20*2*sonar_range)/map_resolution))

    # intensity_map, probability_map, observed_swaths, range_map = @timeit to "map_generation_opt" generate_map_optimized!(
    #     n_rows, n_colums, n_bins, map_resolution, 
    #     map_origin, swath_locals, sonar_range, 
    #     sonar_horizontal_beam_spread, swath_ground_resolution,
    #     knn_k, knn_max_dist, knn_max_variance,
    #     probability_map, intensity_map, range_map, altitude_map,
    #     observed_swaths, probabilities, intensities, altitudes, ranges,
    #     cell_transformations, intensity_variance, 
    #     cell_visited, cells_to_visit, indexes, 
    #     cell_coordinates, intensity_values, to
    # )

    # # Original method
    # knn_k = 4
    # knn_max_dist = 0.2
    # knn_max_variance = 0.05

    # probability_map = ones(n_rows, n_colums)
    # intensity_map = fill(NaN, (n_rows, n_colums))
    # range_map = fill(NaN, (n_rows, n_colums))
    # altitude_map = fill(NaN, (n_rows, n_colums))
    # intensity_variance = fill(NaN,n_rows,n_colums)

    # buffer_size = length(swath_locals)

    # observed_swaths = Array{Vector{Int}}(undef, n_rows, n_colums)

    # for i=1:n_rows, j=1:n_colums
    #     observed_swaths[i,j] = fill(-1, buffer_size)
    # end

    # observed_swaths_cell = zeros(Int, buffer_size)
    # probabilities = zeros(Float64, buffer_size)
    # intensities = zeros(Float64, buffer_size) 
    # altitudes = zeros(Float64, buffer_size) 
    # ranges = zeros(Float64, buffer_size) 

    # cell_coordinates = fill(SVector(0.0,0.0), n_rows*n_colums)
    # intensity_values = zeros(n_rows*n_colums)

    # cell_local = Array{SVector{2,Float64}}(undef, 2, 2)

    # intensity_map, probability_map, observed_swaths, range_map = @timeit to "map_generation_org" generate_map_original!(
    #     n_rows, n_colums, n_bins, map_resolution, 
    #     map_origin, swath_locals, sonar_range, 
    #     sonar_horizontal_beam_spread, swath_ground_resolution,
    #     knn_k, knn_max_dist, knn_max_variance,
    #     probability_map, intensity_map, range_map, altitude_map,
    #     observed_swaths, intensity_variance, cell_local, 
    #     observed_swaths_cell, probabilities, intensities, 
    #     altitudes, ranges, cell_coordinates, intensity_values, to
    # )
    
    # # Raw knn
    # knn_k = 4
    # knn_max_dist = 0.3
    # knn_max_variance = 0.05

    # intensity_map = fill(NaN, (n_rows, n_colums))
    # intensity_variance = fill(NaN,n_rows,n_colums)

    # bin_coordinates = fill(SVector(0.0,0.0), 2 * n_bins * length(swath_locals))
    # intensity_values = zeros(2 * n_bins * length(swath_locals))

    # observed_swaths = fill(Int[], (n_rows, n_colums))
    # range_map = fill(NaN, (n_rows, n_colums))

    # intensity_map, intensity_variance = @timeit to "map_generation_knn" generate_map_knn!(
    #     n_rows, n_colums, n_bins, map_resolution, 
    #     map_origin, swath_ground_resolution, knn_k, knn_max_dist, knn_max_variance,
    #     swath_locals, intensity_map, intensity_variance, bin_coordinates, intensity_values, to)

    show(to)

    # intensity_map = speckle_reducing_bilateral_filter(intensity_map, 0.1)
    # intensity_map = speckle_reducing_bilateral_filter(intensity_map, 0.3)
    # intensity_map = speckle_reducing_bilateral_filter(intensity_map, 0.5)

    # echo_intensity_map = bilateral_filter(echo_intensity_map, 0.1, 0.5)
    # echo_intensity_map = bilateral_filter(echo_intensity_map, 0.3, 0.7)
    # echo_intensity_map = bilateral_filter(echo_intensity_map, 0.5, 0.9)

    return intensity_map, probability_map, observed_swaths, range_map
    # return intensity_map, intensity_variance, observed_swaths, range_map
end
 

function generate_map_optimized!(n_rows, n_colums, n_bins, map_resolution, 
    map_origin, swath_locals, sonar_range, 
    sonar_horizontal_beam_spread, swath_ground_resolution,
    knn_k, knn_max_dist, knn_max_variance,
    probability_map, intensity_map, range_map, altitude_map,
    observed_swaths, probabilities, intensities, altitudes, ranges,
    cell_transformations, intensity_variance, 
    cell_visited, cells_to_visit, indexes, 
    cell_coordinates, intensity_values, to)

    @timeit to "swath_iteration" begin

        for (swath, swath_index) in zip(swath_locals, 0:length(swath_locals)-1) # 0 indexed

            @timeit to "swath_allocation" begin

                # Array containing the polar coordinates of the cell corners relative the the current measurement
                cell_transformations = fill!(cell_transformations, SVector(NaN, NaN)) 
                cell_visited = fill!(cell_visited, false) 

                # Find cell map coordinates of the sonar base
                v = swath.odom[1:2] - map_origin
                row = Int(ceil(-v[1] / map_resolution))
                colum = Int(ceil(v[2] / map_resolution))

                empty!(cells_to_visit)

                # Do 25 connectivity for first cell (8 is not sufficient)
                for i in -2:2, j in -2:2
                    push!(cells_to_visit, SVector(row+i, colum+j))
                end  
            end

            @timeit to "cell_iteration" begin

                while !isempty(cells_to_visit)
                    row, colum = pop!(cells_to_visit)

                    if cell_visited[row ,colum]
                        continue
                    end

                    cell_visited[row ,colum] = true

                    @timeit to "cell_meas_trans" calculate_cell_measurement_transformation!(
                        cell_transformations, row, colum, swath.odom, map_origin, map_resolution
                    )

                    prob_observation = @timeit to "cell_probability" get_cell_probability_gaussian(
                        cell_transformations, row, colum, sonar_range, sonar_horizontal_beam_spread
                    ) :: Float64

                    if prob_observation >= 0.1
                        intensity = @timeit to "cell_intensity" get_cell_intensity(
                            cell_transformations, row, colum, swath, 
                            swath_ground_resolution, n_bins
                        ) :: Float64

                        @timeit to "vector_pushing" begin
                            if !isnan(intensity)
                                indexes[row, colum] += 1
                                # observed_swaths[row, colum][indexes[row,colum]] = swath_index # 0 indexed
                                probabilities[row, colum][indexes[row,colum]] = prob_observation
                                intensities[row, colum][indexes[row,colum]] = intensity
                                # altitudes[row, colum][indexes[row,colum]] = swath.altitude
                                # ranges[row, colum][indexes[row,colum]] = Statistics.mean(
                                #     view(cell_transformations, row:row+1, colum:colum+1))[1]
                            end
                        end

                        @timeit to "find_new_cells" begin
                            for i=1:4
                                new_cell = SVector{2,Int}(row,colum) + four_nn[i]
                                if checkbounds(Bool, cell_visited, new_cell[1], new_cell[2])
                                    push!(cells_to_visit, new_cell)
                                end
                            end
                        end
                    end 
                end
            end
        end
    end

    @timeit to "map_iteration" begin
        for row=1:n_rows, colum=1:n_colums

            index = indexes[row,colum]

            if index == 0
                continue
            end
            
            intensity_map[row, colum] = dot(
                view(intensities[row, colum], 1:index),
                view(probabilities[row, colum], 1:index) / 
                sum(view(probabilities[row, colum], 1:index))
            )

            # probability_map[row, colum] *= prod(1 .- view(probabilities[row, colum], 1:index))
            # range_map[row, colum] = Statistics.mean(view(ranges[row, colum], 1:index))
            # altitude_map[row, colum] = Statistics.mean(view(altitudes[row, colum], 1:index))
        end
    end

    @timeit to "map_interpolation" knn_filtering!(
            n_rows, n_colums, map_resolution, knn_k, knn_max_dist, knn_max_variance,
            intensity_map, intensity_variance, cell_coordinates, intensity_values
        )
            
    return intensity_map, probability_map, observed_swaths, range_map
end


function generate_map_optimized_new!(n_rows, n_colums, n_bins, map_resolution, 
    map_origin, swath_locals, sonar_range, 
    sonar_horizontal_beam_spread, swath_slant_resolution,
    knn_k, knn_max_dist, knn_max_variance,
    probability_map, intensity_map, range_map, altitude_map,
    observed_swaths, probabilities, intensities, altitudes, ranges,
    cell_transformations, intensity_variance, 
    cell_visited, cells_to_visit, indexes, cell_coordinates, intensity_values, 
    sonar_theta, sonar_alpha, sonar_x_offset, sonar_y_offset, sonar_z_offset, to)

    @timeit to "swath_iteration" begin

        max_length_buffer = length(cells_to_visit)
        println(max_length_buffer)

        for (swath, swath_index) in zip(swath_locals, 0:length(swath_locals)-1) # 0 indexed

            @timeit to "swath_allocation" begin

                # Array containing the polar coordinates of the cell corners relative the the current measurement
                cell_transformations = fill!(cell_transformations, SVector(NaN, NaN)) 
                cell_visited = fill!(cell_visited, false) 

                # Find cell map coordinates of the sonar base
                v = swath.odom[1:2] - map_origin
                row = Int(ceil(-v[1] / map_resolution))
                colum = Int(ceil(v[2] / map_resolution))
        
                corrected_altitude_port = ((swath.altitude - sonar_z_offset) * 
                                          cos(swath.odom[3]) * cos(swath.odom[4])) + 
                                          sonar_x_offset * sin(swath.odom[4]) +
                                          sonar_y_offset * sin(swath.odom[3])
                corrected_altitude_stb = ((swath.altitude - sonar_z_offset) * 
                                         cos(swath.odom[3]) * cos(swath.odom[4])) + 
                                         sonar_x_offset * sin(swath.odom[4]) -
                                         sonar_y_offset * sin(swath.odom[3])

                first_bottom_return_range_port = corrected_altitude_port / 
                                                 sin(sonar_theta + sonar_alpha/2 - swath.odom[3])
                first_bottom_return_range_stb = corrected_altitude_stb / 
                                                sin(sonar_theta + sonar_alpha/2 + swath.odom[3])

                horisontal_y_offset = sonar_y_offset * cos(swath.odom[3]) +
                                      sonar_z_offset * sin(swath.odom[3])

                empty!(cells_to_visit)

                # Do 25 connectivity for first cell (8 is not sufficient)
                for i in -2:2, j in -2:2
                    push!(cells_to_visit, SVector(row+i, colum+j))
                end  
            end

            @timeit to "cell_iteration" begin

                while !isempty(cells_to_visit)
                    row, colum = pop!(cells_to_visit)

                    if cell_visited[row ,colum]
                        continue
                    end

                    cell_visited[row ,colum] = true

                    @timeit to "cell_meas_trans" calculate_cell_measurement_transformation!(
                        cell_transformations, row, colum, swath.odom, map_origin, map_resolution
                    )

                    prob_observation = @timeit to "cell_probability" get_cell_probability_gaussian(
                        cell_transformations, row, colum, sonar_range, sonar_horizontal_beam_spread
                    ) :: Float64

                    if prob_observation >= 0.05
                        intensity = @timeit to "cell_intensity" get_cell_intensity_non_corr_swath(
                            cell_transformations, row, colum, 
                            swath, swath_slant_resolution, n_bins, horisontal_y_offset,
                            corrected_altitude_port, corrected_altitude_stb,
                            first_bottom_return_range_port, first_bottom_return_range_stb
                        ) :: Float64

                        @timeit to "vector_pushing" begin
                            if !isnan(intensity)
                                indexes[row, colum] += 1
                                # observed_swaths[row, colum][indexes[row,colum]] = swath_index # 0 indexed
                                probabilities[row, colum][indexes[row,colum]] = prob_observation
                                intensities[row, colum][indexes[row,colum]] = intensity
                                # altitudes[row, colum][indexes[row,colum]] = swath.altitude
                                # ranges[row, colum][indexes[row,colum]] = Statistics.mean(
                                #     view(cell_transformations, row:row+1, colum:colum+1))[1]
                            end
                        end

                        @timeit to "find_new_cells" begin
                            for i=1:4
                                new_cell = SVector{2,Int}(row,colum) + four_nn[i]
                                if checkbounds(Bool, cell_visited, new_cell[1], new_cell[2])
                                    push!(cells_to_visit, new_cell)
                                end
                            end
                        end
                        max_length_buffer = max(max_length_buffer, length(cells_to_visit))
                    end 
                end
            end
        end
    end

    println(max_length_buffer)

    @timeit to "map_iteration" begin
        for row=1:n_rows, colum=1:n_colums

            index = indexes[row,colum]

            if index == 0
                continue
            end
            
            intensity_map[row, colum] = dot(
                view(intensities[row, colum], 1:index),
                view(probabilities[row, colum], 1:index) / 
                sum(view(probabilities[row, colum], 1:index))
            )

            # probability_map[row, colum] *= prod(1 .- view(probabilities[row, colum], 1:index))
            # range_map[row, colum] = Statistics.mean(view(ranges[row, colum], 1:index))
            # altitude_map[row, colum] = Statistics.mean(view(altitudes[row, colum], 1:index))
        end
    end

    # @timeit to "map_interpolation" knn_filtering!(
    #         n_rows, n_colums, map_resolution, knn_k, knn_max_dist, knn_max_variance,
    #         intensity_map, intensity_variance, cell_coordinates, intensity_values
    #     )
            
    return intensity_map, probability_map, observed_swaths, range_map
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
                            swath, swath_resolution, n_bins)

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


function get_cell_intensity_non_corr_swath(cell_transformations, row, colum, 
    swath, swath_slant_resolution, n_bins, horisontal_y_offset,
    corrected_altitude_port, corrected_altitude_stb,
    first_bottom_return_range_port, first_bottom_return_range_stb)

    pixel_intensity = 0.0
    valid_corners = 0

    corrected_altitude = (cell_transformations[row, colum][2] < pi) ?
                         corrected_altitude_stb :
                         corrected_altitude_port

    first_bottom_return_range = (cell_transformations[row, colum][2] < pi) ?
                                first_bottom_return_range_stb : 
                                first_bottom_return_range_port

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


function generate_map_original!(n_rows, n_colums, n_bins, map_resolution, 
    map_origin, swath_locals, sonar_range, 
    sonar_horizontal_beam_spread, swath_ground_resolution,
    knn_k, knn_max_dist, knn_max_variance,
    probability_map, intensity_map, range_map, altitude_map,
    observed_swaths, intensity_variance, cell_local, 
    observed_swaths_cell, probabilities, intensities, 
    altitudes, ranges, cell_coordinates, intensity_values, to)

    cell_global = SVector{2, Float64}

    @timeit to "cell_iteration" begin
        for row=1:n_rows, col=1:n_colums
            @timeit to "cell_alloc" begin
                cell_global = SVector(
                    map_origin[1] - (row - 1) * map_resolution,
                    map_origin[2] + (col - 1) * map_resolution
                ) 
                probability = 1.0
                prob_observation = NaN
                index = 0
            end

            @timeit to "swath_iteration" begin

                for (swath, swath_number) in zip(swath_locals, 0:length(swath_locals)-1) # 0 indexed

                    @timeit to "cell_meas_trans" get_cell_coordinates!(
                        cell_local, cell_global, swath.odom, map_resolution
                    )
                    prob_observation = @timeit to "cell_probability" get_cell_probability_gaussian(
                        cell_local, sonar_range, sonar_horizontal_beam_spread
                    ) :: Float64

                    if prob_observation >= 0.1
                        intensity = @timeit to "cell_intensity" get_cell_echo_intensity(
                            cell_local, swath, swath_ground_resolution, n_bins
                        ) :: Float64

                        if !isnan(intensity)
                            index += 1
                            intensities[index] = intensity
                            probabilities[index] = prob_observation
                            altitudes[index] = swath.altitude
                            ranges[index] = Statistics.mean(cell_local)[1]
                            observed_swaths_cell[index] = swath_number
                            probability *= (1 - prob_observation)
                        end
                    end
                end
            end

            @timeit to "cell_calculation" begin
                if index > 0
                    intensity_map[row, col] = dot(
                        view(intensities, 1:index),
                        view(probabilities, 1:index) / sum(view(probabilities, 1:index))
                    )
                    probability_map[row,col] = probability
                    range_map[row,col] = Statistics.mean(view(ranges, 1:index))
                    altitude_map[row,col] = Statistics.mean(view(altitudes, 1:index))
                    observed_swaths[row,col] = view(observed_swaths_cell, 1:index)
                end
            end
        end
    end

    @timeit to "map_interpolation" knn_filtering!(
            n_rows, n_colums, map_resolution, knn_k, knn_max_dist, knn_max_variance,
            intensity_map, intensity_variance, cell_coordinates, intensity_values
        )
    
    return intensity_map, probability_map, observed_swaths, range_map
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

function generate_map_knn!(n_rows, n_colums, n_bins, map_resolution, 
    map_origin ::SVector{2,Float64}, swath_ground_resolution, knn_k, knn_max_dist, knn_max_variance,
    swath_locals, intensity_map, intensity_variance, bin_coordinates, intensity_values, to)

    @timeit to "setup_data_vectors" begin
        n_data_points = 0

        for swath in swath_locals
            @timeit to "setup_swath" begin

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
                    @timeit to "setup_bin" begin
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
            end
        end
    end

    kdtree = @timeit to "build_kd_tree" NearestNeighbors.KDTree(
        view(bin_coordinates, 1:n_data_points), 
        Distances.Euclidean()
    )

    idx = Vector{Int}(undef, knn_k)
    dist = Vector{Float64}(undef, knn_k)
    svals = SVector{knn_k, Float64}

    @timeit to "generate_map" begin
        for row=1:n_rows, col=1:n_colums

            @timeit to "generate_cell" begin 
                @timeit to "knn" NearestNeighbors.knn_point!(
                    kdtree,
                    map_origin + SVector{2,Float64}(-(row-1)*map_resolution, (col-1)*map_resolution),
                    false, dist, idx, always_false
                )

                @timeit to "cell_calculation" begin
                    svals = view(intensity_values, idx[dist .<= knn_max_dist])

                    if length(svals) > 0
                        var = Statistics.var(svals)
                        if var <= knn_max_variance
                            intensity_map[row,col] = Statistics.mean(svals)
                            intensity_variance[row,col] = var
                        else
                            intensity_map[row,col] = Statistics.quantile!(svals, 10/100)
                            intensity_variance[row,col] = knn_max_variance
                        end
                    end

                end
            end
        end
    end
    
    return intensity_map, intensity_variance
end

function knn_filtering!(n_rows, n_colums, map_resolution, knn_k, knn_max_dist, knn_max_variance,
    intensity_map, intensity_variance, cell_coordinates, intensity_values)
    
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
        if !isnan(intensity_map[row,col])
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
                # intensity_variance[row,col] = var
            else
                intensity_map[row,col] = Statistics.quantile(svals, 10/100)
                # intensity_variance[row,col] = knn_max_variance
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