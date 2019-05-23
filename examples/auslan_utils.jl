using CSV
using DataFrames

lcols = Dict(
    :x => :Column1,
    :y => :Column2,
    :z => :Column3,
    :roll => :Column4,
    :pitch => :Column5,
    :yaw => :Column6,
    :thumb_bend => :Column7,
    :forefinger_bend => :Column8,
    :middlefinger_bend => :Column9,
    :ringfinger_bend => :Column10,
    :littlefinger_bend => :Column11,
    )
rcols = Dict(
    :x => :Column12,
    :y => :Column13,
    :z => :Column14,
    :roll => :Column15,
    :pitch => :Column16,
    :yaw => :Column17,
    :thumb_bend => :Column18,
    :forefinger_bend => :Column19,
    :middlefinger_bend => :Column20,
    :ringfinger_bend => :Column21,
    :littlefinger_bend => :Column22,
    )

function filename(sign, folder_index, trial; folder_base = "tctodd", root = "examples/", ext = ".tsd")
    return string(root, folder_base, "/", folder_base, folder_index, "/", sign, "-", trial, ext)
end

function get_data(sign_str, cols, hand, t = 1:2:30)
    Nfiles, Ntrials, index = 9, 3, 1
    N = Nfiles*Ntrials
    data = Dict(k => zeros(length(t), N) for k in cols)
    for f = 1:Nfiles, j=1:Ntrials
        file = filename(sign_str, f, j)
        df = CSV.File(file, delim="\t", header=false) |> DataFrame
        for k=cols
            data[k][:, index] .= df[hand[k]][t]
        end
        index += 1
    end
    return data
end

