module MapGenerationFunctions

using StaticArrays
using LinearAlgebra
using Rotations

# Struct to represent the polar coordinates of each of the four corners, referenced to a local frame.
# First element contains the range and second thetha.  
struct CellLocal
    tl_corner::SVector{2,Float64}
    tr_corner::SVector{2,Float64}
    br_corner::SVector{2,Float64}
    bl_corner::SVector{2,Float64}
end

function generate_map(n_rows, n_colums, n_bins, map_resolution, map_origin_x, map_origin_y, measurements, sonar_range, sonar_alpha)
    # Variation of Burguera et al. 2016, Algorithm 2

    map_origin = SVector{2,Float64}(map_origin_x, map_origin_y)
    probability_map = zeros(Float64,n_rows,n_colums)
    echo_intensity_map = zeros(Float64,n_rows,n_colums)

    for row=1:n_rows, col=1:n_colums

        observed_measurements = []
        local_cell_coordinates = []


        for measurement in measurements

            measurement_origin = SVector{3,Float64}(
                measurement.odom.pose.pose.position.x,
                measurement.odom.pose.pose.position.y,
                0.0
            )

            cell_l = get_cell_coordinates(
                row, col, measurement_origin, map_resolution, map_origin
            )
            if is_cell_observed(cell_l, sonar_range, sonar_alpha)
                observed_measurements.append(measurement)
                local_cell_coordinates.append(cell_l)
            end
        end

        probability = 1
        echo_intensity = 0
        for (measurement, cell_l) in zip(observed_measurements, local_cell_coordinates)
            prob_measurement = get_cell_probability_uniform(cell_l, sonar_alpha)
            echo_intensity_measurement = get_cell_echo_intensity(
                cell_l, measurement, measurement_resolution, n_bins
            )

            probability *= (1 - prob_measurement)
            echo_intensity += prob_measurement * echo_intensity_measurement
        end

        probability_map[row,col] = probability
        echo_intensity_map[row,col] = echo_intensity
    end

    return echo_intensity_map, probability_map
end

# Returns the polar coordinates of all four cell corners referenced the local frame (SE(3)).
# Assumes that the world coordinate system is NED
function get_cell_coordinates(row, colum, local_frame, map_resolution, map_origin)
    cell_global_x = map_origin[1] - (row - 1) * map_resolution
    cell_global_y = map_origin[2] - (colum - 1) * map_resolution 
    cell_global = SVector(cell_global_x,cell_global_y)

    v1 = cell_global - local_frame[1:2]
    v2 = cell_global - local_frame[1:2] + SVector(0,map_resolution)
    v3 = cell_global - local_frame[1:2] + SVector(-map_resolution,map_resolution) 
    v4 = cell_global - local_frame[1:2] + SVector(-map_resolution,0)
    a1 = rem2pi((local_frame[3] + atan(v1[2], v1[1])), RoundNearest)
    a2 = rem2pi((local_frame[3] + atan(v2[2], v2[1])), RoundNearest)
    a3 = rem2pi((local_frame[3] + atan(v3[2], v3[1])), RoundNearest)
    a4 = rem2pi((local_frame[3] + atan(v4[2], v4[1])), RoundNearest)

    return CellLocal(SVector(norm(v1),a1),SVector(norm(v2),a2),SVector(norm(v3),a3),SVector(norm(v4),a4))
end

function is_cell_observed(cell_l, sonar_range, sonar_alpha)

    min_r = min(
        cell_l.tl_corner[1], 
        cell_l.tr_corner[1], 
        cell_l.br_corner[1], 
        cell_l.bl_corner[1]
    )

    min_theta = min(
        abs(cell_l.tl_corner[2] - pi/2), 
        abs(cell_l.tr_corner[2] - pi/2), 
        abs(cell_l.br_corner[2] - pi/2), 
        abs(cell_l.bl_corner[2] - pi/2),
        abs(cell_l.tl_corner[2] - pi*3/2), 
        abs(cell_l.tr_corner[2] - pi*3/2), 
        abs(cell_l.br_corner[2] - pi*3/2), 
        abs(cell_l.bl_corner[2] - pi*3/2)
    )

    if min_r < sonar_range && min_theta < sonar_alpha
        return true
    else
        return false
    end
end
    
function get_cell_probability_uniform(cell_l, sonar_alpha)

    # Correct if cell is part of left or right swath
    if cell_l.tl_corner[2] > pi
        correction = pi*3/2
    else
        correction = pi/2
    end

    min_theta = min(
        abs(cell_l.tl_corner[2] - correction), 
        abs(cell_l.tr_corner[2] - correction), 
        abs(cell_l.br_corner[2] - correction), 
        abs(cell_l.bl_corner[2] - correction)
    )

    max_theta = max(
        abs(cell_l.tl_corner[2] - correction), 
        abs(cell_l.tr_corner[2] - correction), 
        abs(cell_l.br_corner[2] - correction), 
        abs(cell_l.bl_corner[2] - correction)
    )

    return (max_theta - min_theta) / sonar_alpha 
end

function get_cell_echo_intensity(cell_l, measurement, measurement_resolution, n_bins)

    intensity = 0::Float64

    for f in fieldnames(typeof(cell_l))
        corner = getfield(cell_l,f)

        # Interpolate
        lower_index = floor(Int64, corner[1]/measurement_resolution)
        higher_index = ceil(Int64, corner[1]/measurement_resolution)
        w1 = corner_r/measurement_resolution - lower_index
        w2 = 1 - w1

        if corner[2] > pi
            intensity += w1*measurement(higher_index+n_bins) + w2*measurement(lower_index+n_bins)
        else
            intensity += w1*measurement(n_bins-higher_index) + w2*measurement(n_bins-lower_index)
        end
    end

    return intensity/4
end

#precompile(knn, (Float64, Vector{Float64}, Vector{Float64},))

end