using DataFrames
using MLDataUtils
using Distributions
using RDatasets
using CSV
using Distributed


addprocs(parse(Int, ENV["SLURM_CPUS_PER_TASK"]))


techniques = ["bow", "tfidf", "glove"]
targets = ["location", "stroke"]

for technique in techniques
    for seed in 1:5
    	for target in targets
	    cd("/nfs/sloanlab001/projects/edema-partners_proj/clean_data/$(target)")
	    print("OK")
	    test_X = CSV.read("X_test_$(technique)_$(seed).csv", header = true)
	    cd("/nfs/sloanlab001/projects/edema-partners_proj/Trees/Outputs/$(target)")
	    lnr = OptimalTrees.readjson("OCTH_auc_cluster_500_$(seed)_$(technique).json")
	    proba = OptimalTrees.predict_proba(lnr, test_X)
	    proba = proba[2]
	    proba = DataFrame(probabilities = proba)
	    cd("/nfs/sloanlab001/projects/edema-partners_proj/proba_$(target)")
	    CSV.write("proba_$(technique)_OCTH_$(seed).csv", proba, writeheader = false)
	end
    end
end

target_b = ["acuity"]
for technique in techniques
    for seed in 1:5
        for target in target_b
            cd("/nfs/sloanlab001/projects/edema-partners_proj/clean_data/$(target)")
            print("OK")
            test_X = CSV.read("X_test_$(technique)_$(seed).csv", header = true)
            cd("/nfs/sloanlab001/projects/edema-partners_proj/Trees/Outputs/$(target)")
            lnr = OptimalTrees.readjson("OCTH_auc_cluster_250_$(seed)_$(technique).json")
            proba = OptimalTrees.predict_proba(lnr, test_X)
            proba = proba[2]
	    proba = DataFrame(probabilities = proba)
	    cd("/nfs/sloanlab001/projects/edema-partners_proj/proba_$(target)")
            CSV.write("proba_$(technique)_OCTH_$(seed).csv", proba,writeheader = false)
        end
    end
end