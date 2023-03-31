module Slam

# add more julia processes
using Distributed
nprocs() < 3 ? addprocs(4-nprocs()) : nothing

using Caesar
Distributed.@everywhere using Caesar

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

filename = "/home/repo/Navigation-brov2/images/pose_and_landmarks_training_data.pickle"

timesteps = load_pickle(filename)

sigma_x = 0.01          # Variance for odometry measurements in x-direction
sigma_y = 0.01          # Variance for odometry measurements in y-direction
sigma_yaw = 0.01        # Variance for odometry measurements in yaw
sigma_r = 0.01          # Variance for range measurements of landmarks
sigma_b = 0.01          # Variance for bearing measurements of landmarks

dist_thresh = 3.0       # Threshold of mahalanobis distance for accepting new landmark or solve for existing

n_sample_points = 10    # Number of sample points for montecarlo simulation


fg = initfg()

meas_information_matrix = [[1/sigma_r, 0] [0, 1/sigma_b]]

n_landmarks = 0

last_data = undef

for (timestep, data) in enumerate(timesteps)

    new_pose = Symbol("x$(timestep-1)")

    addVariable!(fg, new_pose, Pose2, tags = [:POSE])

    if timestep == 1
        pp2 = PriorPose2(MvNormal(
            [data.pose[1]; data.pose[2]; data.pose[5]], 
            Matrix(Diagonal([0.1;0.1;0.1].^2))
        ))
        addFactor!(fg, [new_pose], pp2)
    else
        p2 = Pose2Pose2(MvNormal(
            [
                data.pose[1] - last_data.pose[1],
                data.pose[2] - last_data.pose[2],
                data.pose[5] - last_data.pose[5] # Is this correct for a rotation?
            ],
            diagm([sigma_x, sigma_y, sigma_yaw].^2)
        ))
        addFactor!(fg, [new_pose, Symbol("x$(timestep)")], p2)
    end

    # Do we need to solve graph here?
    solveTree!(fg)

    # Vector that contains dictionaries of all landmarks in range of a measurement 
    # with the associated measurement marginal for the landmark ( p(z|d,Z^-) )
    landmarks_to_assosiate = [Dict{Symbol,Float64}() for _ in 1:length(data.measurements)]

    landmarks = ls(fg, tags=[:LANDMARK;])

    n_landmarks = length(landmarks)

    # Find all landmarks that are "in range" of each measurements and add new landmark if non in range
    for (idx, measurement) in enumerate(data.measurements)

        for landmark in landmarks

            dist = Distances.mahalanobis(
                getPPE(getVariable(landmark)), 
                [measurement.bearing, measurement.range], 
                meas_information_matrix
            )

            if dist < dist_thresh
                landmarks_to_assosiate[idx][landmark] = 0.0 
            end
        end

        if isempty(landmarks_to_assosiate[idx])
            
            new_landmark = Symbol("l$n_landmarks")

            landmarks_to_assosiate[idx][new_landmark] = 0.0 

            addVariable!(fg, new_landmark, Point2, tags = [:LANDMARK])

            # landmark_bearing = data.measurements[idx].bearing
            # landmark_range = data.measurements[idx].range
            
            # p2br = Pose2Point2BearingRange(Normal(landmark_bearing, sigma_b),Normal(landmark_range, sigma_r))
            # addFactor!(fg, [new_pose, new_landmark], p2br)
            
            n_landmarks += 1

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

            for sample_point in sample_points

                meas = [
                    data.measurements[idx].x, # Needs to be range bearing (not global) to be correct
                    data.measurements[idx].y
                ]

                pose_evaluation_pose = [[sample_point[1], sample_point[2], acos(sample_point[3])] [1.0,1.0,1.0]]

                landmark_evaluation_point = [[sample_point[1] + data.measurements[idx].x, sample_point[1] + data.measurements[idx].x] [1.0,1.0]]

                println(new_pose_mdf(pose_evaluation_pose))
                # Something iffy here! Returns only NaN, probably not initialized
                println(landmark_mdf(landmark_evaluation_point))

                p = landmark_mdf(landmark_evaluation_point)[1]

                if isnan(p)
                    p = 0.0
                end

                prob += new_pose_mdf(pose_evaluation_pose)[1] * p

            end

            landmarks_to_assosiate[idx][landmark] = prob

        end
    end

    # Find marginalized assosiation probability for all possible assosiations
    for (idx, measurement) in enumerate(data.measurements)

        # Find assosiation probabilities for each possible assosiation for current measurement
        # and create a multimodal factor

        probabilities = Float64[]

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

        variables = collect(keys(landmarks_to_assosiate[idx]))
        
        insert!(variables, 1, new_pose)
        insert!(probabilities, 1, 1.0)

        println(probabilities)
        println(variables)

        # Do we need some normalization of probabilities?
        println(sum(probabilities))

        landmark_bearing = data.measurements[idx].bearing
        landmark_range = data.measurements[idx].range
            
        p2br = Pose2Point2BearingRange(Normal(landmark_bearing, sigma_b),Normal(landmark_range, sigma_r))

        addFactor!(fg, variables, p2br, multihypo=probabilities)

    end

    last_data = data

    solveTree!(fg)

end

end # module Slam