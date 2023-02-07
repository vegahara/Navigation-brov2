module MapGeneration

using StaticArrays

function generate_map()
    
    # generate inverse probabilistic map

    # generate echo intensity map

    # use method to build geometric map


function generate_probabilistic_map(n_rows, n_colums, map_resolution, map_origin_x, map_origin_y, measurements)

    map_origin[map_origin_x, map_origin_y]  isa SVector{2, Float64}

    for row=1:n_rows, col=1:n_colums
        cell_global_x, cell_global_y = map_cell_to_global_coordinate(row, col, map_resolution, map_origin_x, map_origin_y)
        for measurement in measurements

    end
end
    
function map_cell_to_global_coordinate(row, colum, map_resolution, map_origin_x, map_origin_y)
        
    cell_global_x = map_origin_x - (row - 1) * map_resolution
    cell_global_y = map_origin_y - (colum - 1) * map_resolution 

    return cell_global_x, cell_global_y
end

function is_cell_observed()