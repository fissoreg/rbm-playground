using MLDatasets
using Images
using ImageIO
using Revise
using Plots
using ArgParse

using LinearAlgebra
using RBMS

# Logging
using TensorBoardLogger, Logging, StatsBase
using RBMS: sample_hiddens, dΘ

# GPU hacks ###################################################################
# following lines are needed for GPU execution.
# TODO: how to get rid of the following tricks?
import RBMS: pseudolikelihood
# let's force CUDA compilation for `exp` on `Float32`.
exp(2f0)
# we need to fix the pseudolikelihood definition...
tmp = rand(5, 5) |> gpu
pseudolikelihood(rbm::AbstractRBM{Float32, V, H}, x::typeof(tmp)) where {V, H} = 0
				
# MNIST utilities and loading ##################################################
flatten(x) = reshape(x, :, size(x)[end])

function mnist(;T::Type=Float32)
	train_x, _ = MNIST.traindata()
	test_x,  _  = MNIST.testdata()

	prepare = flatten ∘ x -> T.(x)

	prepare(train_x), prepare(test_x)
end

function mnist_to_img(samples; c=10, r=10, h=28, w=28)
	f = zeros(r*h,c*w)
	for i=1:r, j=1:c
		f[(i-1)*h+1:i*h,(j-1)*w+1:j*w] = reshape(samples[:,(i-1)*c+j],h,w)'
	end
	w_min = minimum(samples)
	w_max = maximum(samples)
	scale = x -> (x-w_min)/(w_max-w_min)
	map!(scale,f,f)
	colorview(Gray,f)
end

# Logging ######################################################################
function make_hist(data)
	hist = fit(Histogram, data[:])
	hist.edges[1], hist.weights
end

function MSRE(rbm, batch)
	mean(norm.(batch - CDk(rbm, batch, 1)[3]))
end

function get_tb_logger(;dir="tensorboard_logs/run", test_data=Nothing)

	lg = TBLogger(dir; step_increment=0)

	function tb_logger(rbm, X, epoch, epoch_time, total_time)
		batch = test_data[:, 1:1000]
		n_pre = 20
		pl = pseudolikelihood(rbm, batch)
		msre = MSRE(rbm, batch)

		W, vb, hb = make_hist.([rbm.W, rbm.vbias, rbm.hbias])
		#U, s, Vt = svd(rbm.W)

		samples_img(init, k) = mnist_to_img(CDk(rbm, init, k)[3])
		rand_init = rand(size(X, 1), 100) |> typeof(X)

		with_logger(lg) do
			@info "Pseudolikelihood" pl log_step_increment=1
			@info "MSRE" msre
			@info "Weights" W=W vbias=vb hbias=hb

			@info "Features" Features=mnist_to_img(rbm.W)

			@info "Samples" reconstructions=samples_img(batch, 1)
			@info "Samples" test_10_steps=samples_img(batch, 10)
			@info "Samples" test_100_steps=samples_img(batch, 100)
			@info "Samples" test_1000_steps=samples_img(batch, 1000)
			@info "Samples" rand_1_step=samples_img(rand_init, 1)
			@info "Samples" rand_10_step=samples_img(rand_init, 10)
			@info "Samples" rand_100_step=samples_img(rand_init, 100)
			@info "Samples" rand_1000_step=samples_img(rand_init, 1000)

			@info "Preactivations" preactivations=mnist_to_img(
				sample_hiddens(rbm, batch[:, 1:n_pre]);
				h=size(rbm.W, 2), w=1, c=n_pre, r=1
			)'

			@info "Gradients" grads=make_hist.(dΘ(rbm, CDk(rbm, batch, 1)))
			#@info "SVs" s[1] s[2] s[3]

			# NOTE: drawback of the iterators interface!!!!! Here we would like
			# a `gradient` function to call
			# @info "Gradients"
		end

		rbm, X, epoch, epoch_time, total_time
	end

	tb_logger
end

# parsing command-line args ####################################################

function parse_commandline()
	s = ArgParseSettings()

	@add_arg_table! s begin
	    "--n_epochs"
	        help = "number of epochs of training"
	        arg_type = Int
		default = 10
	    "--lr", "--learning_rate", "-α"
	        help = "learning rate"
		arg_type = Float32
	        default = 0.01f0
	    "--batch_size", "--bs"
	        help = "size of the minibatches"
		arg_type = Int
		default = 10
	    "--k", "-k"
	    	help = "number of sampling iterations (e.g. the k in CDk)"
		arg_type = Int
		default = 1
	    "--hidden_nodes", "--nh"
	        help = "number of hidden nodes"
		arg_type = Int
		default = 100
	end

	to_named_tuple(d) = (; zip(keys(d), values(d))...)
	parse_args(ARGS, s, as_symbols=true) |> to_named_tuple
end

args = parse_commandline()

# hyperparameters ##############################################################
n_epochs = 100

α = 0.1f0
batch_size = 100
k = 1
nh = 100

# training #####################################################################
train_x, test_x = mnist() .|> gpu
d, n = size(train_x)

logger = (args -> console_logger(args...)) ∘ get_tb_logger(;
	dir="tensorboard/CD/k=$(args.k)_nh=$(args.hidden_nodes)_α=$(args.lr)_" *
	    "bs=$args.batch_size",
	test_data=test_x,
)

# Define model and use GPU if available
rbm = RBM(Float32, Unitary, Bernoulli, d, nh; X=train_x) |> gpu

RBMS.fit!(
	rbm, train_x;
	n_epochs=args.n_epochs,
	logger=logger,
	batch_size=args.batch_size,
	α=args.lr,
	k=args.k,
)
