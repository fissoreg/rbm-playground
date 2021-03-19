### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 9ae9f102-7791-11eb-33bc-d9f83f2c790a
begin
	using MLDatasets
	using Images
	using ImageIO
	using Revise
	using Plots
	
	using LinearAlgebra
	using RBMS

	function ingredients(path::String)
		# this is from the Julia source code (evalfile in base/loading.jl)
		# but with the modification that it returns the module instead of the last object
		name = Symbol(basename(path))
		m = Module(name)
		Core.eval(m,
			Expr(:toplevel,
				 :(eval(x) = $(Expr(:core, :eval))($name, x)),
				 :(include(x) = $(Expr(:top, :include))($name, x)),
				 :(include(mapexpr::Function, x) = $(Expr(:top, :include))(mapexpr, $name, x)),
				 :(include($path))))
		m
	end
end

# ╔═╡ 15de4708-7792-11eb-1303-4756712e2209
begin
	flatten(x) = reshape(x, :, size(x)[end])
	
	function mnist(;T::Type=Float64)
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
	
	train_x, test_x = mnist()
	d, n = size(train_x)
end

# ╔═╡ ee6f97c2-7ffd-11eb-290e-8528cc56befb
begin
	using TensorBoardLogger, Logging, StatsBase
	using RBMS: sample_hiddens, dΘ
	
	function make_hist(data)
  		hist = fit(Histogram, data[:])
  		hist.edges[1], hist.weights
	end

	function MSRE(rbm, batch)
		mean(norm.(batch - CDk(rbm, batch, 1)[3]))
	end
	
	function get_tb_logger(;dir="tensorboard_logs/run")
		
		lg = TBLogger(dir; step_increment=0)
	
		function tb_logger(rbm, X, epoch, epoch_time, total_time)
			batch = test_x[:, 1:1000]
			n_pre = 20
			pl = pseudolikelihood(rbm, batch)
			msre = MSRE(rbm, batch)
			
			W, vb, hb = make_hist.([rbm.W, rbm.vbias, rbm.hbias])
			#U, s, Vt = svd(rbm.W)
			
			samples_img(init, k) = mnist_to_img(CDk(rbm, init, k)[3])
			rand_init = rand(size(X, 1), 100)

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
end

# ╔═╡ b64464f6-7aba-11eb-0278-bddb0797fde9
begin
	n_epochs = 100
	
	αs = [0.1]
	bss = [100]
	ks = [1]
	nhs = [100]

	for α in αs
		for bs in bss
			for k in ks
				for nh in nhs
					logger = (args -> console_logger(args...)) ∘ get_tb_logger(;
						dir="tensorboard/CD/k=$(k)_nh=$(nh)_α=$(α)_bs=$bs"
					)

					rbm = RBM(Float64, Unitary, Bernoulli, d, nh; X=train_x)

					RBMS.fit!(
						rbm, train_x;
						n_epochs=n_epochs,
						logger=logger,
						batch_size=bs,
						α=α,
						k=k,
					)
				end
			end	
		end
	end
end

# ╔═╡ 3725add0-8026-11eb-3fb8-a7c974a88ead
U,s,Vt = svd(rbm.W)

# ╔═╡ cb88dc66-8020-11eb-16f3-f9e0a4944f3e
begin
	pyplot()
	plot(rbm.W[:])
	histogram(rbm.vbias)
end

# ╔═╡ Cell order:
# ╠═9ae9f102-7791-11eb-33bc-d9f83f2c790a
# ╠═15de4708-7792-11eb-1303-4756712e2209
# ╠═ee6f97c2-7ffd-11eb-290e-8528cc56befb
# ╠═02236b56-802e-11eb-113a-8b0cd37f213c
# ╠═b64464f6-7aba-11eb-0278-bddb0797fde9
# ╠═3725add0-8026-11eb-3fb8-a7c974a88ead
# ╠═cb88dc66-8020-11eb-16f3-f9e0a4944f3e
# ╠═1a674304-7c62-11eb-1476-c52fa298e487
