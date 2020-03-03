#Pkg.add("DataFrames")
#Pkg.add("CSV")
#Pkg.add("OptimalTrees")
#Pkg.add("CSV")
#Pkg.add("RDatasets")
#Pkg.add("Distributions")



#using OptimalTrees
using DataFrames
using MLDataUtils
using Distributions
using RDatasets
using CSV
using Distributed


addprocs(parse(Int, ENV["SLURM_CPUS_PER_TASK"]))
#addprocs(8)
cd("/nfs/sloanlab001/projects/edema-partners_proj/Trees/Outputs/stroke")

function RunOCTH(technique, maxDepth, minBucket, seed, test_X, test_y, train_X, valid_X, train_y, valid_y)	

	lnr = OptimalTrees.OptimalTreeClassifier(ls_num_tree_restarts = 250,fast_num_support_restarts = 250, ls_max_hyper_iterations = 250, ls_num_hyper_restarts = 250, ls_random_seed = seed)
        grid = OptimalTrees.GridSearch(lnr, Dict(
    :max_depth => 2:maxDepth:2,
    :minbucket => 1:minBucket:2,
    :criterion => [:misclassification, :gini, :entropy],
    :hyperplane_config => ((sparsity=:05,),(sparsity=:10,),(sparsity=:20,),(sparsity=:30,),(sparsity=:40,),(sparsity=:50,))
        ))
	OptimalTrees.fit!(grid, train_X, train_y, valid_X, valid_y, validation_criterion = :auc)
	grid.lnr	
	println(grid.best_params)
	println(grid.lnr)
	
	cd("/nfs/sloanlab001/projects/edema-partners_proj/Trees/Outputs/stroke")
        OptimalTrees.writedot("OCTH_auc_cluster_500_$(seed)_$(technique).dot", grid.lnr)
        run(`dot -Tpng -o OCTH_auc_cluster_500_$(seed)_$(technique).png OCTH_auc_cluster_500_$(seed)_$(technique).dot`)
	OptimalTrees.writejson("OCTH_auc_cluster_500_$(seed)_$(technique).json", grid.lnr; write_tree = true)
		# OptimalTrees.fit!(grid, train_X, train_y, valid_X, valid_y)
	misclass = OptimalTrees.score(grid.lnr, test_X, test_y, criterion=:misclassification)
	auc = OptimalTrees.score(grid.lnr, test_X, test_y, criterion=:auc)

	println("The cross-validated OCTH misclassification rate is ", misclass)
	println("The cross-validated OCTH auc is ", auc)

	return OptimalTrees.predict(grid.lnr, test_X), auc
end

techniques = ["bow", "tfidf", "glove"]

for technique in techniques
	df_b = DataFrame()
	for seed in 1:5
	    	cd("/nfs/sloanlab001/projects/edema-partners_proj/clean_data/stroke")
		#params = CSV.read("params$(seed)_$(technique).csv", header = true);

		maxDepth = 10;
		minBucket = 10;
		seed = seed;
		technique = technique;

		test_X = CSV.read("X_test_$(technique)_$(seed).csv", header = true);
		test_y = CSV.read("y_test_$(technique)_$(seed).csv", header = false);
		big_X = CSV.read("X_train_$(technique)_$(seed).csv", header = true);
		big_y = CSV.read("y_train_$(technique)_$(seed).csv", header = false);
		println(nobs(big_X))
		println(nobs(big_y))
		println(nobs(convert(DataFrame,big_X)))
		(train_X, train_y), (valid_X, valid_y) = IAIBase.splitobs(:classification, big_X, big_y, train_proportion=0.67)
		#(train_X,train_y), (valid_X,valid_y) = splitobs(shuffleobs((convert(DataFrame,big_X), big_y)), at = 0.67)
		#valid_X = CSV.read("X_val$(seed)_$(technique).csv", header = true);
		#valid_y = CSV.read("y_val$(seed)_$(technique).csv", header = true);

		pred_y, auc = RunOCTH(technique, maxDepth, minBucket, seed, test_X[:,1:end], test_y[:,1],train_X[:,1:end],valid_X[:,1:end], train_y[:,1],valid_y[:,1])
		
		cd("/nfs/sloanlab001/projects/edema-partners_proj/Trees/Outputs/stroke")
		df_t = vcat(pred_y,auc)
		df_b = hcat(df_b,df_t, makeunique=true)
		println(df_b)
	end
	println(technique)
	CSV.write("OCTH_technique_auc_cluster_500_$(technique).csv", df_b)
end









