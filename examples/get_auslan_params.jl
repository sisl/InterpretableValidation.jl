include("auslan_utils.jl")
using GaussianProcesses

t = 1:2:30 # Define the timeteps to grab
ks = keys(hand)
Nfiles = 1#9
Ntrials = 1#3

function get_all_data(signs, cols, hand = rcols)
    N = Nfiles*Ntrials*length(signs)*length(cols)

    # Load in all examples for the sign of my choice
    data = zeros(length(t), N)
    index = 1
    for f = 1:Nfiles, j=1:Ntrials, s=signs
        file = filename(s, f, j)
        df = CSV.File(file, delim="\t", header=false) |> DataFrame
        for c in cols
            data[:, index] .= df[hand[c]][t]
            index += 1
        end
    end
    return data
end

signs = ["alive", "cold", "come", "drink", "go", "hear", "I", "lose", "more", "paper", "read", "surprise", "take", "us", "voluntary", "why", "yes", "zero"]
bend_cols = [:thumb_bend, :forefinger_bend, :middlefinger_bend, :ringfinger_bend, :littlefinger_bend]
angle_cols = [:roll, :pitch, :yaw]
motion_cols = [:x, :y, :z]

data = get_all_data(signs, bend_cols)
gp.kernel
gp = GP([1.:15...], data[:,1], MeanZero(), SE(0.0,0.0))
optimize!(gp, noise=false)
gp.kernel.σ2
gp.kernel.ℓ2

σ2_avg = 0
ℓ2_avg = 0
N = size(data,2)
for i=1:N

    gp = GP([1.:15...], data[:,i], MeanZero(), SE(0.0,0.0))
    optimize!(gp)
    global σ2_avg += gp.kernel.σ2
    if gp.kernel.ℓ2 < 225
        global ℓ2_avg += gp.kernel.ℓ2
    end
end
σ2_avg /= N
ℓ2_avg /= N

sqrt(13)


xtest = 1.:15
samps = rand(GP(Float64[], Float64[], MeanZero(), SE(log(3^2),log(σ2_avg))), xtest, 5)

using Plots
plot(xtest, samps)
scatter!(data[:,1])

