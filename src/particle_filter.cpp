/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <math.h>
#include <numeric>
#include <random>

#include "particle_filter.h"

using std::default_random_engine;
using std::normal_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  default_random_engine gen;
  normal_distribution<double> N_x_init(x, std[0]);
  normal_distribution<double> N_y_init(y, std[1]);
  normal_distribution<double> N_theta_init(theta, std[2]);

  num_particles = 100;
  is_initialized = true;

  weights.clear();
  particles.clear();
  for (int i = 0; i < num_particles; ++i) {
    weights.push_back(1);
    particles.push_back(Particle({
        i, N_x_init(gen), N_y_init(gen), N_theta_init(gen), 1
    }));
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  default_random_engine gen;
  normal_distribution<double> N_x(0, std_pos[0]);
  normal_distribution<double> N_y(0, std_pos[1]);
  normal_distribution<double> N_yaw(0, std_pos[2]);

  // For each particle, calculate the new value based on the bicycle motion model, and then
  // add random process noise to it.
  for (int i = 0; i < num_particles; ++i) {
    if (fabs(yaw_rate) >= 0.00001) {
      double theta_rate = particles[i].theta + yaw_rate * delta_t;
      particles[i].x += velocity / yaw_rate * (sin(theta_rate) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(theta_rate));
      particles[i].theta = theta_rate;
    } else {
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    }
    particles[i].x += N_x(gen);
    particles[i].y += N_y(gen);
    particles[i].theta += N_yaw(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  // Not exactly sure what this function is supposed to do, ignoring...
}

void ParticleFilter::updateWeights(
    double sensor_range, double std_landmark[], 
    std::vector<LandmarkObs> observations, Map map_landmarks) {
  double multivar_denom_outside = 1. / (2. * M_PI * std_landmark[0] * std_landmark[1]);
  double multivar_denom_inside_x = 2. * std_landmark[0] * std_landmark[0];
  double multivar_denom_inside_y = 2. * std_landmark[1] * std_landmark[1];
  weights.clear();

  // For each particle:
  for (Particle& particle : particles) {
    // Process each observation.
    double log_weight = 0;
    for (const LandmarkObs& obs : observations) {
      // First, transform the observations to the map coordinate system.
      // http://planning.cs.uiuc.edu/node99.html (Eq 3.33)
      LandmarkObs new_obs;
      new_obs.x = obs.x * cos(particle.theta) - obs.y * sin(particle.theta) + particle.x;
      new_obs.y = obs.x * sin(particle.theta) + obs.y * cos(particle.theta) + particle.y;

      // Then, for each observation, find the nearest landmark.
      Map::single_landmark_s best_landmark;
      double min_distance = DBL_MAX;
      for (const auto& landmark : map_landmarks.landmark_list) {
        double new_dist = dist(landmark.x_f, landmark.y_f, new_obs.x, new_obs.y);
        if (new_dist < min_distance) {
          min_distance = new_dist;
          best_landmark = landmark;
        }
      }

      // Then, compute the new weight for this observation.
      log_weight += log(multivar_denom_outside *
          exp(-pow(best_landmark.x_f - new_obs.x, 2) / multivar_denom_inside_x -
              pow(best_landmark.y_f - new_obs.y, 2) / multivar_denom_inside_y));
    }
    particle.weight = exp(log_weight);
    weights.push_back(particle.weight);
  }
}

void ParticleFilter::resample() {
  default_random_engine gen;
  std::discrete_distribution<int> distribution(weights.begin(), weights.end());
  std::vector<Particle> new_particles;
  std::vector<double> new_weights;
  for (int i = 0; i < num_particles; ++i) {
    int ind = distribution(gen);
    new_particles.push_back(particles[ind]);
    new_weights.push_back(particles[ind].weight);
  }
  particles = new_particles;
  weights = new_weights;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
