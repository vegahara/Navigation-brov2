module Slam

# add more julia processes
using Distributed
nprocs() < 4 ? addprocs(4-nprocs()) : nothing

using Caesar
Distributed.@everywhere using Caesar

import Distances
using RoMEPlotting
using PyCall

py"""
import pickle
import sys
sys.path.append('utility_functions')
import utility_functions
 
def load_pickle(filename):
    with open(filename, 'rb') as f:
        objects = []
        while True:
            try:
                obj = pickle.load(f)
                objects.append(obj)
            except EOFError:
                break
    return objects
"""

load_pickle = py"load_pickle"

filename = "/home/repo/Navigation-brov2/images/landmark_detection/pose_and_landmarks_training_data.pickle"

timesteps = load_pickle(filename)

# Threshold to determine which landmark hypotheses to assosiate. 
#If no landmarks is above the threshold, a new landmark is created
landmark_likelihood_treshold = 1e-6  

n_sample_points = 50    # Number of sample points for montecarlo simulation

fg = initfg()

last_data = undef

for (timestep, data) in enumerate(timesteps)

    new_pose = Symbol("x$(timestep)")

    addVariable!(fg, new_pose, Pose2, tags = [:POSE])

    if timestep == 1
        pp2 = PriorPose2(MvNormal(
            [data.pose[2]; data.pose[1]; rem2pi(-data.pose[5] + pi/2, RoundNearest)], 
            Matrix(Diagonal(
                [data.pose[6][1]; data.pose[6][8]; data.pose[6][36]]
            ))
        ))

        addFactor!(fg, [new_pose], pp2)
    else
        # Transform from world to body NED (2D)
        T_w_b = [
            cos(data.pose[5])  sin(data.pose[5]) 0.0;
            -sin(data.pose[5]) cos(data.pose[5]) 0.0;
            0.0 0.0 1.0
        ]

        # Transform from body NED (2D) to body SE2
        # We flip the y coordinate and since we here are using relative 
        # yaw angles, only the sign has to be flipped for yaw
        T_b_b = [
            1.0 0.0  0.0;
            0.0 -1.0  0.0;
            0.0 0.0  -1.0
        ]

        T = T_b_b * T_w_b

        # For some odd reason we need to calculate the transform 
        # from the new pose and BACKWARDS to the old pose. 
        Δx_w = -[
            data.pose[1] - last_data.pose[1],
            data.pose[2] - last_data.pose[2],
            data.pose[5] - last_data.pose[5],
        ]

        Δx_b = T * Δx_w

        Δx_b[3] = rem2pi(Δx_b[3], RoundNearest)

        Σ_w = Matrix(Diagonal([
            data.pose[6][1] - last_data.pose[6][1],
            data.pose[6][8] - last_data.pose[6][8],
            data.pose[6][36]
        ]))

        Σ_b = T * Σ_w * T'
        
        # Handels numerical errors in calculation of Σ_b
        sΣ_b = Symmetric(Σ_b)

        if count(>(1e-14),abs.(Σ_b - sΣ_b)) > 0
            throw(ErrorException("Transformed covariance matrix is not positive definite"))
        end

        println(sΣ_b)

        p2 = Pose2Pose2(MvNormal(Δx_b, sΣ_b))

        addFactor!(fg, [new_pose, Symbol("x$(timestep-1)")], p2)
    end

    doautoinit!(fg, new_pose)

    # Vector that contains dictionaries of all landmarks candidate hypotheses
    # and their likelihoods for one measurement (p(z|d,Z^-))
    landmarks_to_assosiate = [Dict{Symbol,Float64}() for _ in 1:length(data.measurements)]

    landmarks = ls(fg, tags=[:LANDMARK;])
    n_landmarks = length(landmarks)

    new_pose_mdf = getBelief(fg, new_pose) # Marginal density function of the new pose
    sample_points = rand(new_pose_mdf, n_sample_points)

    # Find landmark candidate hypotheses above likelihood threshold
    for (meas_idx, measurement) in enumerate(data.measurements)

        pose_evaluation_poses = []
        landmark_evaluation_points = []

        for sample_point in sample_points

            yaw = atan(sample_point[4],sample_point[3]) # Convert from rotation matrix to yaw
        
            if isempty(pose_evaluation_poses)
                pose_evaluation_poses = hcat([sample_point[1], sample_point[2], yaw])
                landmark_evaluation_points = hcat([
                    sample_point[1] + data.measurements[meas_idx].range * cos(yaw - data.measurements[meas_idx].bearing), 
                    sample_point[2] + data.measurements[meas_idx].range * sin(yaw - data.measurements[meas_idx].bearing)
                ])
            else
                pose_evaluation_poses = hcat(pose_evaluation_poses, [sample_point[1], sample_point[2], yaw])
                landmark_evaluation_points = hcat(landmark_evaluation_points, [
                    sample_point[1] + data.measurements[meas_idx].range * cos(yaw - data.measurements[meas_idx].bearing), 
                    sample_point[2] + data.measurements[meas_idx].range * sin(yaw - data.measurements[meas_idx].bearing)
                ])
            end
        end

        for landmark in landmarks

            landmark_mdf = getBelief(fg, landmark)
            likelihood = (
                sum(new_pose_mdf(pose_evaluation_poses) 
                .* landmark_mdf(landmark_evaluation_points)) 
                / n_sample_points
            )

            if likelihood >= landmark_likelihood_treshold
                landmarks_to_assosiate[meas_idx][landmark] = likelihood 
            end
        end

        # Create new landmark if no candidate hypotheses above likelihood threshold
        if isempty(landmarks_to_assosiate[meas_idx])

            n_landmarks += 1
            
            new_landmark = Symbol("l$(n_landmarks)")

            addVariable!(fg, new_landmark, Point2, tags = [:LANDMARK])

            landmark_bearing = data.measurements[meas_idx].bearing
            landmark_range = data.measurements[meas_idx].range
                
            p2br = Pose2Point2BearingRange(
                Normal(rem2pi(-landmark_bearing, RoundNearest), measurement.sigma_b),
                Normal(landmark_range, measurement.sigma_r)
            )

            addFactor!(fg, [new_pose, new_landmark], p2br)

            doautoinit!(fg, new_landmark)
        end
    end
    
    # Find assosiation probabilities for each landmark candidate hypothesis for each measurement
    # and (potentially) create a multimodal data assosiation factor
    for (meas_idx, measurement) in enumerate(data.measurements)

        # No candidate hypotheses for measurement (a new landmark has been created earlier for measurement)
        if isempty(landmarks_to_assosiate[meas_idx])
            continue
        end

        variables = collect(keys(landmarks_to_assosiate[meas_idx]))
        insert!(variables, 1, new_pose)

        # No need to calculate assisiation probabilities for only one landmark
        if length(variables) == 2

            landmark_bearing = data.measurements[meas_idx].bearing
            landmark_range = data.measurements[meas_idx].range
                
            p2br = Pose2Point2BearingRange(
                Normal(rem2pi(-landmark_bearing, RoundNearest), measurement.sigma_b),
                Normal(landmark_range, measurement.sigma_r)
            )
            
            addFactor!(fg, variables, p2br)

            continue
        end

        # Calculate assosiation probabilities
        probabilities = Float64[]

        for (current_assosiation_hyp, _) in landmarks_to_assosiate[meas_idx]

            prob = 0.0

            # Try all permutations of assosiations for the other measurements
            for assosiations in Iterators.product(landmarks_to_assosiate...)

                # Brute force method of keeping current assosiation of current measurement fixed
                if assosiations[meas_idx].first != current_assosiation_hyp
                    continue
                end

                # Remove assosiations that try to assign two different measurement to one landmark
                if length(union(assosiations)) < length(assosiations)
                    continue
                end

                inner_prob = 1.0

                for ass in assosiations
                    inner_prob *= ass.second
                end

                prob += inner_prob
            end

            push!(probabilities, prob)

        end

        # Normalizing and inserting probability for new pose
        probabilities = probabilities ./ sum(probabilities)
        insert!(probabilities, 1, 1)

        if any(isnan, probabilities)
            @warn "Not able to assosiate measurement"
            continue
        end
        
        landmark_bearing = data.measurements[meas_idx].bearing
        landmark_range = data.measurements[meas_idx].range
            
        p2br = Pose2Point2BearingRange(
            Normal(rem2pi(-landmark_bearing, RoundNearest), measurement.sigma_b),
            Normal(landmark_range, measurement.sigma_r)
        )

        addFactor!(fg, variables, p2br, multihypo=probabilities)
    end

    global last_data = data

    solveTree!(fg)

    # p3 = plotSLAM2D(fg, dyadScale=1.0, drawPoints=false, drawTriads=true, drawEllipse=false, levels=3)

    # p3 |> Gadfly.PDF("/home/repo/Navigation-brov2/images/slam/2D_plot.pdf")

    saveDFG("/home/repo/Navigation-brov2/images/slam/factor_graphs/fg_x$(timestep)", fg)

end


# solveTree!(fg)

# p3 = plotSLAM2D(fg, dyadScale=1.0, drawPoints=false, drawTriads=false, drawEllipse=false, levels=3)

# p3 |> Gadfly.PDF("/home/repo/Navigation-brov2/images/full_training_200_swaths/2D_plot.pdf")

# p2 = drawGraph(fg)
    
# p2 |> Gadfly.PDF("/home/repo/Navigation-brov2/images/landmark_detection_data/training_100_swaths/graph_plot.pdf")

end # module Slam