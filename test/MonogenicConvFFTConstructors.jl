@testset "Monogenic ConvFFT constructors" begin
	# random initialization
	originalSize = (20, 16, 1, 10);
	mono = MonogenicLayer(originalSize, scale = 4, σ = abs, Monotype = GaussianLP(), averagingLayer = false);
	@test mono.σ == abs
end