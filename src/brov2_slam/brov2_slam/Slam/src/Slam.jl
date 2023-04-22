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

filename = "/home/repo/Navigation-brov2/images/full_training_200_swaths/pose_and_landmarks_training_data.pickle"

timesteps = load_pickle(filename)

sigma_x = 0.5          # Variance for odometry measurements in x-direction
sigma_y = 0.5          # Variance for odometry measurements in y-direction
sigma_yaw = 0.1        # Variance for odometry measurements in yaw
sigma_r = 1.0          # Variance for range measurements of landmarks
sigma_b = 0.1          # Variance for bearing measurements of landmarks

dist_thresh = 5.0       # Threshold of mahalanobis distance for accepting new landmark or solve for existing

n_sample_points = 10    # Number of sample points for montecarlo simulation

fg = initfg()

meas_information_matrix = [[1/sigma_r, 0] [0, 1/sigma_b]]

n_landmarks = 0

last_data = undef

for (timestep, data) in enumerate(timesteps)

    new_pose = Symbol("x$(timestep)")

    addVariable!(fg, new_pose, Pose2, tags = [:POSE])

    if timestep == 1
        pp2 = PriorPose2(MvNormal(
            [data.pose[2]; data.pose[1]; rem2pi(-data.pose[5] - pi/2, RoundNearest)], 
            Matrix(Diagonal([0.1;0.1;0.01].^2))
        ))

        addFactor!(fg, [new_pose], pp2)
    else
        x_diff_w = data.pose[1] - last_data.pose[1]
        y_diff_w = data.pose[2] - last_data.pose[2]

        p2 = Pose2Pose2(MvNormal(
            [
                x_diff_w * cos(data.pose[5]) + y_diff_w * sin(data.pose[5]), # Transform from world NED to body SE2
                x_diff_w * sin(data.pose[5]) - y_diff_w * cos(data.pose[5]), # Transform from world NED to body SE2
                rem2pi((data.pose[5] - last_data.pose[5]), RoundNearest)
            ],
            diagm([sigma_y, sigma_x, sigma_yaw].^2)
        ))

        addFactor!(fg, [new_pose, Symbol("x$(timestep-1)")], p2)
    end

    doautoinit!(fg, new_pose)

    # Vector that contains dictionaries of all landmarks in range of a measurement 
    # with the associated measurement marginal for the landmark ( p(z|d,Z^-) )
    landmarks_to_assosiate = [Dict{Symbol,Float64}() for _ in 1:length(data.measurements)]

    landmarks = ls(fg, tags=[:LANDMARK;])

    n_landmarks = length(landmarks)

    # Find all landmarks that are "in range" of each measurements and add new landmark if non in range
    current_pose = getPPEMax(fg, new_pose)

    println(current_pose)

    for (idx, measurement) in enumerate(data.measurements)

        for landmark in landmarks

            landmark_pose = getPPEMax(fg, landmark)

            landmark_pose_trans = landmark_pose - current_pose[1:2]
            
            range_landmark = sqrt(sum((landmark_pose_trans).^2))

            bearing_landmark = rem2pi(
                (current_pose[3] + pi) - atan(landmark_pose_trans[2], landmark_pose_trans[1]),
                RoundNearest
            )

            dist = Distances.mahalanobis(
                [rem2pi(bearing_landmark, RoundNearest), range_landmark], 
                [rem2pi(measurement.bearing, RoundNearest), measurement.range], 
                meas_information_matrix
            )

            # println([rem2pi(bearing_landmark, RoundNearest), range_landmark])
            # println([rem2pi(measurement.bearing, RoundNearest), measurement.range])
            # println(dist)

            if dist < dist_thresh
                landmarks_to_assosiate[idx][landmark] = 0.0 
            end
        end

        if isempty(landmarks_to_assosiate[idx])

            n_landmarks += 1
            
            new_landmark = Symbol("l$n_landmarks")

            #landmarks_to_assosiate[idx][new_landmark] = 0.0 

            addVariable!(fg, new_landmark, Point2, tags = [:LANDMARK])

            landmark_bearing = data.measurements[idx].bearing
            landmark_range = data.measurements[idx].range
                
            p2br = Pose2Point2BearingRange(
                Normal(rem2pi(pi - landmark_bearing, RoundNearest), sigma_b),
                Normal(landmark_range, sigma_r)
            )

            addFactor!(fg, [new_pose, new_landmark], p2br)

            doautoinit!(fg, new_landmark)
            
            continue
        end
    end

    # Find p(z|d,Z^-) for all possible assosiations each of the measurements
    new_pose_mdf = getBelief(fg, new_pose)

    for (idx, assosiations) in enumerate(landmarks_to_assosiate)
        for (landmark, _) in assosiations

            landmark_mdf = getBelief(fg, landmark)
            sample_points = rand(new_pose_mdf, n_sample_points)
            prob = 0.0

            println(landmark)
            println([data.measurements[idx].bearing, data.measurements[idx].range])

            for sample_point in sample_points

                # println(current_pose)
                # println([sample_point[1], sample_point[2], atan(sample_point[4],sample_point[3])])
                # println([sample_point[1], sample_point[2], -acos(sample_point[3])])

                yaw = atan(sample_point[4],sample_point[3])

                pose_evaluation_pose = [[sample_point[1], sample_point[2], yaw] [1.0,1.0,1.0]]

                landmark_evaluation_point = [[sample_point[1] + data.measurements[idx].range * cos(yaw + pi - data.measurements[idx].bearing), sample_point[2] + data.measurements[idx].range * sin(yaw + pi - data.measurements[idx].bearing)] [1.0,1.0]]

                println(new_pose_mdf(pose_evaluation_pose))
                # Something iffy here! Returns only NaN, probably not initialized
                println(landmark_mdf(landmark_evaluation_point))

                p = landmark_mdf(landmark_evaluation_point)[1]

                if isnan(p)
                    p = 0.0
                else
                    landmark_pose = getPPEMax(fg, landmark)

                    println(current_pose)

                    println(landmark_pose)
                    println(landmark_evaluation_point[1:2])
                end

                prob += new_pose_mdf(pose_evaluation_pose)[1] * p

            end

            println(prob)

            landmarks_to_assosiate[idx][landmark] = prob

        end
    end

    # Find marginalized assosiation probability for all possible assosiations
    for (idx, measurement) in enumerate(data.measurements)

        # Find assosiation probabilities for each possible assosiation for current measurement
        # and create a multimodal factor

        probabilities = Float64[]

        if isempty(landmarks_to_assosiate[idx])
            continue
        end

        for (ass_of_current_meas, _) in landmarks_to_assosiate[idx]

            prob = 0.0

            # Try all permutations of assosiations for the other measurements
            for assosiations in Iterators.product(landmarks_to_assosiate...)
                # Brute force method of keeping current assosiation of current measurement fixed
                if assosiations[idx].first != ass_of_current_meas
                    continue
                end
                # Can not assign two different measurement to one landmark
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

        # Normalizing
        println(probabilities)
        probabilities = probabilities ./ sum(probabilities)

        variables = collect(keys(landmarks_to_assosiate[idx]))

        println(probabilities)
        println(variables)

        # Remove highly unlikely assosiations
        for i in length(variables):-1:1
            if (probabilities[i] < 1.0e-10) || isnan(probabilities[i])
                deleteat!(probabilities, i)
                deleteat!(variables, i)
            end
        end

        # Is this a viable solution???
        if isempty(probabilities)
            continue
        end
        
        insert!(variables, 1, new_pose)
        insert!(probabilities, 1, 1)

        landmark_bearing = data.measurements[idx].bearing
        landmark_range = data.measurements[idx].range
            
        p2br = Pose2Point2BearingRange(
            Normal(rem2pi(pi - landmark_bearing, RoundNearest), sigma_b),
            Normal(landmark_range, sigma_r)
        )

        if length(variables) > 2
            println(probabilities)
            addFactor!(fg, variables, p2br, multihypo=probabilities)
        else
            addFactor!(fg, variables, p2br)
        end
    end

    global last_data = data

    # solveTree!(fg)

    # p3 = plotSLAM2D(fg, dyadScale=1.0, drawPoints=false, drawTriads=false, drawEllipse=false, levels=3)

    # p3 |> Gadfly.PDF("/home/repo/Navigation-brov2/images/full_training_200_swaths/2D_plot.pdf")

end

solveTree!(fg)

p3 = plotSLAM2D(fg, dyadScale=1.0, drawPoints=false, drawTriads=false, drawEllipse=false, levels=3)

p3 |> Gadfly.PDF("/home/repo/Navigation-brov2/images/full_training_200_swaths/2D_plot.pdf")

p2 = drawGraph(fg)
    
p2 |> Gadfly.PDF("/home/repo/Navigation-brov2/images/full_training_200_swaths/graph_plot.pdf")

end # module Slam