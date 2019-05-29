using Plots
include("auslan_utils.jl")

file = filename("you", 2, 1)

df = CSV.File(file, delim="\t", header=false) |> DataFrame
# plot(df[lcols[:x]], label="x - left")
plot(df[rcols[:x]], label="x - right", title = "XYZ")

# plot!(df[lcols[:y]], label="y - left")
plot!(df[rcols[:y]], label="y - right")

# plot!(df[lcols[:z]], label="z - left")
plot!(df[rcols[:z]], label="z - right")


# plot(df[lcols[:x]], label="x - left")
plot(df[rcols[:roll]], label="x - right", title="roll, pitch, yaw")

# plot!(df[lcols[:y]], label="y - left")
plot!(df[rcols[:pitch]], label="y - right")

# plot!(df[lcols[:z]], label="z - left")
plot!(df[rcols[:yaw]], label="z - right")



plot(df[rcols[:thumb_bend]], label="thumb_bend")
plot!(df[rcols[:forefinger_bend]], label="forefinger_bend")
plot!(df[rcols[:middlefinger_bend]], label="middlefinger_bend")
plot!(df[rcols[:ringfinger_bend]], label="ringfinger_bend")
plot!(df[rcols[:littlefinger_bend]], label="littlefinger_bend")

