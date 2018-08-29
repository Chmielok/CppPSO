#pragma once
#include <functional>
#include <vector>
#include <random>


namespace CppPSO {
	struct Boundaries {
		Boundaries(double lower, double upper) :
			lower(lower < upper ? lower : upper),
			upper(lower < upper ? upper : lower) {};

		double width() {
			return upper - lower;
		}

		double lower;
		double upper;
	};

	class dVector : public std::vector<double> {
		using std::vector<double>::vector;
	public:
		dVector() : std::vector<double>() { };
		dVector(const int dimensions, const std::vector<Boundaries>& bounds, std::default_random_engine& randomEngine);
		dVector operator+(const dVector& other) const;
		dVector operator+=(const dVector& other);
		dVector operator-(const dVector& other) const;
		dVector operator*(const dVector& other) const;
		dVector operator*(const double scale) const;
		dVector& clamp(const std::vector<Boundaries>& bounds);
	};

	class PSO
	{
	public:
		struct AlgorithmParams {
			enum PopulationGenerationAlgorithm { RANDOM, EVEN_DISTRIBUTION } pointGenerationAlgorithm;
			int initialPointsPerAxis;
			int iterations;
		};

		PSO(const AlgorithmParams& _params, const std::function<double(std::vector<double>&)>& _evaluatedFunction,
			const std::vector<Boundaries>& _bounds, const int dimensions);
		~PSO();

		/**
		* Runs the algorithm, returns best position and its evaluated value.
		*/
		std::tuple<std::vector<double>, double> run();

	private:
		class Swarm {
		public:
			Swarm(const int _size, const int _dimensions, const std::vector<Boundaries>& bounds,
				const PSO::AlgorithmParams::PopulationGenerationAlgorithm _algorithm, 
				const std::function<double(std::vector<double>&)>& evaluatedFunction);

			void iterate();
			std::tuple<dVector, double> getBestParticle();

		private:
			class Particle {
			public:
				Particle(const dVector& _position, const std::vector<Boundaries>& _bounds, 
					const std::shared_ptr<std::default_random_engine>& randomEngine, 
					const std::function<double(std::vector<double>&)>& _evaluatedFunction);
				double move(dVector& globalBestPosition);
				double getEvaluatedValue() const;
				dVector getPosition() const;

			private:
				void markIfBest();
				dVector calculateMovement(dVector& globalBestPosition);
				std::function<double(std::vector<double>&)> evaluatedFunction;
				int dimensions;
				std::shared_ptr<std::default_random_engine> randomEngine;
				std::vector<Boundaries> bounds;

				dVector position;
				double evalValue;
				dVector velocity;

				dVector bestPosition;
				double bestValue;
			};

			static std::vector<Particle> createPoints(
				const int size, const int dimensions, const std::vector<Boundaries>& bounds,
				const PSO::AlgorithmParams::PopulationGenerationAlgorithm algorithm, 
				const std::shared_ptr<std::default_random_engine>& randomEngine, 
				const std::function<double(std::vector<double>&)>& evaluatedFunction);
			static std::vector<Particle> generateEvenlyDistributedPoints(
				const int size, const int dimensions, const std::vector<Boundaries>& bounds, 
				const std::shared_ptr<std::default_random_engine>& randomEngine,
				const std::function<double(std::vector<double>&)>& evaluatedFunction);
			static std::vector<Particle> generateRandomPoints(
				const int size, const int dimensions, const std::vector<Boundaries>& bounds, 
				const std::shared_ptr<std::default_random_engine>& randomEngine,
				const std::function<double(std::vector<double>&)>& evaluatedFunction);

			std::vector<Particle> particles;
			dVector bestParticlePosition;
			double bestParticleValue;
			int dimensions;
			std::shared_ptr<std::default_random_engine> randomEngine;
		};

		PSO(); // forbidden

		AlgorithmParams params;
		std::vector<Boundaries> bounds;
		std::function<double(std::vector<double>&)> evaluatedFunction;
		int dimensions;
		Swarm swarm;
	};
};