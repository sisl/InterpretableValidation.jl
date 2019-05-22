using CSV
using DataFrames

function filename(sign, trial, folder_index; folder_base = "tctodd", root = "examples/", ext = ".tsd")
    return string(root, folder_base, "/", folder_base, folder_index, "/", sign, "-", trial, ext)
end

filename("drink", 1, 1)
file = "examples/tctodd/tctodd1/drink-1.tsd"


df = CSV.File(file, delim="\t", header=false) |> DataFrame
println(df)

df

