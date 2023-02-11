module MapGenerationFunctions

using StaticArrays
using LinearAlgebra

# Struct to represent the polar coordinates of each of the four corners, referenced to a local frame.
# First element contains the range and second thetha.  
struct CellLocal
    tl_corner::SVector{2,Float64}
    tr_corner::SVector{2,Float64}
    br_corner::SVector{2,Float64}
    bl_corner::SVector{2,Float64}
end

struct Swath
    data_port
    data_stb
    odom::SVector{3,Float64}
    altitude::Float64
end

function generate_map(n_rows, n_colums, n_bins, map_resolution, map_origin_x, map_origin_y, swaths, sonar_range, sonar_alpha, swath_ground_resolution)

    map_origin = SVector{2,Float64}(map_origin_x, map_origin_y)
    probability_map = fill(1.0, (n_rows, n_colums))
    echo_intensity_map = fill(NaN, (n_rows, n_colums))
    
    swath_locals = [Swath(
        swath.data_port, 
        swath.data_stb, 
        SVector{3,Float64}(swath.odom[1],swath.odom[2],swath.odom[3]),
        float(swath.altitude)) 
        for swath in swaths]

    for row=1:n_rows, col=1:n_colums
        cell_global = get_cell_global_coordinates(row,col,map_resolution,map_origin)
        echo_intensity = 0.0
        probability = 1.0
        cell_valid = false
        n_valid_swaths = 0

        for swath in swath_locals

            cell_l = get_cell_coordinates(cell_global, swath.odom, map_resolution)

            if is_cell_observed(cell_l, sonar_range, sonar_alpha)

                prob_swath = get_cell_probability_uniform(cell_l, sonar_alpha)
                echo_intensity_swath, intensity_valid = get_cell_echo_intensity(cell_l, swath, swath_ground_resolution, n_bins)
                
                if intensity_valid
                    cell_valid = true
                    n_valid_swaths += 1
                    echo_intensity += prob_swath * echo_intensity_swath
                    # echo_intensity = echo_intensity * probability * (1 - prob_swath) + prob_swath * echo_intensity_swath
                    probability *= (1 - prob_swath)
                end
            end
        end

        if cell_valid
            probability_map[row,col] = probability
            echo_intensity_map[row,col] = echo_intensity / n_valid_swaths
        end

        if col == n_colums
            println(row)
        end
    end

    return echo_intensity_map, probability_map
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

    a = rem2pi.(atan.([v1[2], v2[2], v3[2], v4[2]], [v1[1], v2[1], v3[1], v4[1]]) .- local_frame[3], RoundDown)

    return CellLocal(SVector(n1, a[1]), SVector(n2, a[2]), SVector(n3, a[3]), SVector(n4, a[4]))
end


function is_cell_observed(cell_l, sonar_range, sonar_alpha)

    min_r = min(
        cell_l.tl_corner[1], 
        cell_l.tr_corner[1], 
        cell_l.br_corner[1], 
        cell_l.bl_corner[1]
    )

    min_theta = minimum(
        abs.([
            cell_l.tl_corner[2] - pi/2, 
            cell_l.tr_corner[2] - pi/2, 
            cell_l.br_corner[2] - pi/2, 
            cell_l.bl_corner[2] - pi/2,
            cell_l.tl_corner[2] - pi*3/2, 
            cell_l.tr_corner[2] - pi*3/2, 
            cell_l.br_corner[2] - pi*3/2, 
            cell_l.bl_corner[2] - pi*3/2
        ])
    )

    return min_r < sonar_range && min_theta < sonar_alpha
end
    

function get_cell_probability_uniform(cell_l, sonar_alpha)

    corners = [cell_l.tl_corner, cell_l.tr_corner, cell_l.br_corner, cell_l.bl_corner]
    correction = pi/2 + pi * (cell_l.tl_corner[2] > pi) * 2

    min_theta = abs.(corners[1][2] - correction)
    max_theta = abs.(corners[1][2] - correction)

    for i = 2:4
        theta = abs(corners[i][2] - correction)
        min_theta = min(min_theta, theta)
        max_theta = max(max_theta, theta)
    end

    if max_theta - min_theta > sonar_alpha
        # Pixel is covering the whole beam and we cant have more than 1 in probability
        return 1
    else
        return (max_theta - min_theta) / sonar_alpha 
    end
end


function get_cell_echo_intensity(cell_l, swath, swath_resolution, n_bins)

    pixel_intensity = 0

    for f in fieldnames(typeof(cell_l))
        corner = getfield(cell_l,f)

        # Interpolate
        lower_index = floor(Int64, corner[1]/swath_resolution) + 1
        higher_index = ceil(Int64, corner[1]/swath_resolution) + 1
        w1 = corner[1]/swath_resolution - lower_index
        w2 = 1 - w1

        # The corner is outside the swath range
        if higher_index > n_bins
            continue
        end

        if corner[2] > pi
            measure_intensity = w1*swath.data_stb[higher_index] + w2*swath.data_stb[lower_index]
        else
            measure_intensity = w1*swath.data_port[1+n_bins-higher_index] + w2*swath.data_port[1+n_bins-lower_index]
        end

        # Do not use pixel if one of the corners evaluate to NaN
        if measure_intensity === NaN
            return NaN, false
        end

        pixel_intensity += measure_intensity
    end

    return pixel_intensity/4, true
end

end
