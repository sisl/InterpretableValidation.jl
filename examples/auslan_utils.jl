using CSV
using DataFrames
using DataStructures
using Statistics

lcols = OrderedDict(
    :x_translation => :Column1,
    :y_translation => :Column2,
    :z_translation => :Column3,
    :roll => :Column4,
    :pitch => :Column5,
    :yaw => :Column6,
    :thumb_bend => :Column7,
    :forefinger_bend => :Column8,
    :middlefinger_bend => :Column9,
    :ringfinger_bend => :Column10,
    :littlefinger_bend => :Column11,
    )
rcols = OrderedDict(
    :x_translation => :Column12,
    :y_translation => :Column13,
    :z_translation => :Column14,
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
    data = OrderedDict(k => zeros(length(t), N) for k in cols)
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

unary_minus(x) = -x
sek(x,xp, l=2, σ2 = 0.2) = σ2*exp(-(x-xp)^2/(2*l^2))

# Write loss function that returns average l2 loss of an expression of a similar size
time_series_loss(data, sample) = mean(sum((data .- sample).^2, dims = 1))

