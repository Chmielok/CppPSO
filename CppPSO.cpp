#include "CppPSO.h"

using namespace CppPSO;
using namespace std;

PSO::PSO(
	const AlgorithmParams& _params,
	const function<double(vector<double>&)>& _evaluatedFunction,
	const vector<Boundaries>& _bounds,
	const int _dimensions) :
	params(_params), evaluatedFunction(_evaluatedFunction), bounds(_bounds), dimensions(_dimensions),
	swarm((int)pow(_params.initialPointsPerAxis, _dimensions), _dimensions, _bounds, _params.pointGenerationAlgorithm, _evaluatedFunction) {};


PSO::~PSO()
{
	// nothing for now
}

tuple<vector<double>, double> PSO::run() {
	for (int it = 0; it < params.iterations; ++it)
		swarm.iterate();
	return swarm.getBestParticle();
}

PSO::Swarm::Swarm(
	const int size, const int _dimensions, const vector<Boundaries>& bounds,
	const PSO::AlgorithmParams::PopulationGenerationAlgorithm algorithm, 
	const function<double(vector<double>&)>& evaluatedFunction) :
	dimensions(_dimensions), 
	randomEngine(new default_random_engine()), 
	bestParticleValue(-INFINITY) {

	particles = createPoints(size, _dimensions, bounds, algorithm, randomEngine, evaluatedFunction);
	for (auto & particle : particles) {
		if (particle.getEvaluatedValue() > bestParticleValue) {
			bestParticlePosition = particle.getPosition();
			bestParticleValue = particle.getEvaluatedValue();
		}
	}
};

void PSO::Swarm::iterate() {
	double iterationBestValue = -INFINITY;
	dVector iterationBestPosition;
	for (auto & particle : particles) {
		double newEvaluatedValue = particle.move(bestParticlePosition);
		if (newEvaluatedValue > iterationBestValue) {
			iterationBestPosition = particle.getPosition();
			iterationBestValue = newEvaluatedValue;
		}
	}
	if (iterationBestValue > bestParticleValue) {
		bestParticlePosition = iterationBestPosition;
		bestParticleValue = iterationBestValue;
	}
}

tuple<dVector, double> PSO::Swarm::getBestParticle() {
	return make_tuple(bestParticlePosition, bestParticleValue);
}

vector<PSO::Swarm::Particle> PSO::Swarm::createPoints(
	const int size, const int dimensions, const vector<Boundaries>& bounds,
	const PSO::AlgorithmParams::PopulationGenerationAlgorithm algorithm, 
	const shared_ptr<default_random_engine>& randomEngine,
	const function<double(vector<double>&)>& evaluatedFunction) {

	switch (algorithm) {
	case PSO::AlgorithmParams::EVEN_DISTRIBUTION:
		return generateEvenlyDistributedPoints(size, dimensions, bounds, randomEngine, evaluatedFunction);
		break;
	case PSO::AlgorithmParams::RANDOM:
		return generateRandomPoints(size, dimensions, bounds, randomEngine, evaluatedFunction);
		break;
	default:
		throw runtime_error("Unknown population generation algorithm.");
	}
}

vector<PSO::Swarm::Particle> PSO::Swarm::generateEvenlyDistributedPoints(
	const int size, const int dimensions, const vector<Boundaries>& bounds, 
	const shared_ptr<default_random_engine>& randomEngine,
	const function<double(vector<double>&)>& evaluatedFunction) {

	// TODO: Implement the logic of this method.
	// Possible solution - create N-dimensional cartesian product and then scale its components to fit between lower-upper boundaries.
	return generateRandomPoints(size, dimensions, bounds, randomEngine, evaluatedFunction);
}

vector<PSO::Swarm::Particle> PSO::Swarm::generateRandomPoints(
	const int size, const int dimensions, const vector<Boundaries>& bounds, 
	const shared_ptr<default_random_engine>& randomEngine,
	const function<double(vector<double>&)>& evaluatedFunction) {

	vector<Particle> particles;
	particles.reserve(size);
	vector<uniform_real_distribution<double>> generators;
	for (int dim = 0; dim < dimensions; ++dim) {
		auto &lowerBound = bounds[dim].lower, &upperBound = bounds[dim].upper;
		generators.push_back(uniform_real_distribution<double>(lowerBound, upperBound));
	}
	for (int i = 0; i < size; ++i) {
		dVector position;
		for (int dim = 0; dim < dimensions; ++dim)
			position.push_back(generators[dim](*randomEngine));
		particles.push_back(Particle(position, bounds, randomEngine, evaluatedFunction));
	}

	return particles;
}

PSO::Swarm::Particle::Particle(
	const dVector& _position, const vector<Boundaries>& _bounds, 
	const shared_ptr<default_random_engine>& _randomEngine,
	const function<double(vector<double>&)>& _evaluatedFunction) :
	position(_position), 
	bounds(_bounds), 
	dimensions(_position.size()),
	velocity(_position.size()),
	evaluatedFunction(_evaluatedFunction), 
	evalValue(_evaluatedFunction(position)), 
	bestPosition(_position), 
	randomEngine(_randomEngine) {

	for (int dim = 0; dim < dimensions; ++dim) {
		double width = bounds[dim].width();
		uniform_real_distribution generator(-width, width);
		velocity[dim] = generator(*randomEngine);
	}
	bestValue = evalValue;
}

double PSO::Swarm::Particle::move(dVector& globalBestPosition) {
	auto oldPosition = position;
	position += calculateMovement(globalBestPosition);
	position.clamp(bounds);
	velocity = position - oldPosition;
	evalValue = evaluatedFunction(position);
	markIfBest();
	return evalValue;
}

double PSO::Swarm::Particle::getEvaluatedValue() const {
	return evalValue;
}

dVector PSO::Swarm::Particle::getPosition() const {
	return position;
}

void PSO::Swarm::Particle::markIfBest() {
	if (evalValue > bestValue) {
		bestPosition = position;
		bestValue = evalValue;
	}
}

dVector PSO::Swarm::Particle::calculateMovement(dVector& globalBestPosition) {
	//TODO: Extract those, so they can be modified.
	Boundaries randomWeightsBoundaries(0, 1);
	double toBestPointWeight = 2, toGlobalBestWeight = 2;
	uniform_real_distribution generator(randomWeightsBoundaries.lower, randomWeightsBoundaries.upper);

	double inertiaFactor = generator(*randomEngine);
	double toBestPointFactor = generator(*randomEngine) * toBestPointWeight;
	double toGlobalBestFactor = generator(*randomEngine) * toGlobalBestWeight;

	dVector inertia = velocity * toGlobalBestFactor;
	dVector toBestPoint = (bestPosition - position) * toBestPointFactor;
	dVector toGlobalBest = (globalBestPosition - position) * toGlobalBestFactor;

	return inertia + toBestPoint + toGlobalBest;
}

dVector::dVector(const int dimensions, const vector<Boundaries>& bounds, default_random_engine& randomEngine) : vector<double>() {
	this->reserve(dimensions);
	for (int dim = 0; dim < dimensions; ++dim) {
		uniform_real_distribution generator(bounds[dim].lower, bounds[dim].upper);
		(*this)[dim] = generator(randomEngine);
	}
}

dVector dVector::operator+(const dVector& other) const {
	dVector result;
	int dimensions = other.size();
	if (size() != dimensions)
		throw runtime_error("Dimensions do not match.");
	result.reserve(dimensions);
	for (int dim = 0; dim < dimensions; ++dim) {
		result.push_back((*this)[dim] + other[dim]);
	}
	return result;
}

dVector dVector::operator+=(const dVector& other) {
	int dimensions = other.size();
	if (size() != dimensions)
		throw runtime_error("Dimensions do not match.");
	for (int dim = 0; dim < dimensions; ++dim) {
		auto & element = (*this)[dim];
		element += other[dim];
	}
	return *this;
}

dVector dVector::operator-(const dVector& other) const {
	dVector result;
	int dimensions = other.size();
	if (size() != dimensions)
		throw runtime_error("Dimensions do not match.");
	result.reserve(dimensions);
	for (int dim = 0; dim < dimensions; ++dim) {
		result.push_back((*this)[dim] - other[dim]);
	}
	return result;
}

dVector dVector::operator*(const double scale) const {
	dVector result;
	int dimensions = size();
	result.reserve(dimensions);
	for (int dim = 0; dim < dimensions; ++dim) {
		result.push_back((*this)[dim] * scale);
	}
	return result;
}

dVector dVector::operator*(const dVector& other) const {
	dVector result;
	int dimensions = other.size();
	if (size() != dimensions)
		throw runtime_error("Dimensions do not match.");
	result.reserve(dimensions);
	for (int dim = 0; dim < dimensions; ++dim) {
		result.push_back((*this)[dim] * other[dim]);
	}
	return result;
}

dVector& dVector::clamp(const vector<Boundaries>& bounds) {
	int dimensions = bounds.size();
	if (size() != dimensions)
		throw runtime_error("Dimensions do not match.");
	for (int dim = 0; dim < dimensions; ++dim) {
		auto & element = (*this)[dim];
		auto & bound = bounds[dim];
		if (element > bound.upper)
			element = bound.upper;
		else if (element < bound.lower)
			element = bound.lower;
	}
	return *this;
}