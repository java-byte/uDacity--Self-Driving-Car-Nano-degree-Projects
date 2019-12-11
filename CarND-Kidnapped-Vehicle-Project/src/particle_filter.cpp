/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles=10;
	
    default_random_engine gen;
    
	// TODO: Set standard deviations for x, y, and theta.
	double std_x, std_y, std_theta;

	std_x=std[0];
	std_y=std[1];
	std_theta=std[2];
    
	// This line creates a normal (Gaussian) distribution for x.
	normal_distribution<double> dist_x(x,std_x);
	normal_distribution<double> dist_y(y,std_y);
	normal_distribution<double> dist_theta(theta,std_theta);

	for(int i =0 ; i<num_particles;i++)
	{
		Particle particle;

		particle.id=i;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		particle.weight=1;

		particles.push_back(particle);
		weights.push_back(particle.weight);
	}

	is_initialized = true;   
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    
	default_random_engine gen;

	double x_f,y_f,theta_f;

	for (int i=0;i<num_particles;i++)
	{
       if(yaw_rate==0)
	   {
		   x_f = particles[i].x + velocity*delta_t*cos(particles[i].theta);
		   y_f = particles[i].y + velocity*delta_t*sin(particles[i].theta);
		   theta_f = particles[i].theta;
	   }
	   
	   else
	   {
		   x_f = particles[i].x + (velocity/yaw_rate)*(sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
		   y_f = particles[i].y + (velocity/yaw_rate)*(cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
		   theta_f = particles[i].theta + yaw_rate*delta_t;
	   }
	   
	   // TODO: Set standard deviations for x, y, and theta.
	double std_x, std_y, std_theta;

	std_x=std_pos[0];
	std_y=std_pos[1];
	std_theta=std_pos[2];
    
	// This line creates a normal (Gaussian) distribution for x.
	normal_distribution<double> dist_x(x_f,std_x);
	normal_distribution<double> dist_y(y_f,std_y);
	normal_distribution<double> dist_theta(theta_f,std_theta);

	particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
	particles[i].theta = dist_theta(gen);
	particles[i].weight=1;

	}


}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
    
// Each observation point is transformed from Car co-ordinate system to Map co-ordinate 	

	for(unsigned int j=0;j<num_particles;j++)
    {
	   vector <LandmarkObs> trans_observation;
	   particles[j].weight=1.0;
	   double multiplier = 1.0;

	for(unsigned int i=0;i< observations.size();i++)
	{
		LandmarkObs trans_obj;
		// transform to map x coordinate
       trans_obj.x = particles[j].x + (cos(particles[j].theta) * observations[i].x) - (sin(particles[j].theta) * observations[i].y);

       // transform to map y coordinate
	   trans_obj.y = particles[j].y + (sin(particles[j].theta) * observations[i].x) + (cos(particles[j].theta) * observations[i].y);
	   
      // Finding the best landmark point associated with transformed point
		double length_min = 99999.0;
		// std::vector<single_landmark_s> landmark_list ;
		double landmark_near_x,landmark_near_y;
		for (unsigned int t=0; t < map_landmarks.landmark_list.size(); t++)
		{
			double landmark_x = map_landmarks.landmark_list[t].x_f;
			double landmark_y = map_landmarks.landmark_list[t].y_f;

			double calc_dist = sqrt((trans_obj.x - landmark_x)*(trans_obj.x - landmark_x) + (trans_obj.y - landmark_y)*(trans_obj.y - landmark_y));

			if(calc_dist < length_min)
			{
				length_min=calc_dist;
				landmark_near_x = landmark_x;
				landmark_near_y = landmark_y;
			}
		}
			double closest_dis = sensor_range;
			// define inputs
			double sig_x = std_landmark[0];
			double sig_y = std_landmark[1];
			double x_obs = trans_obj.x;
			double y_obs = trans_obj.y;
			double mu_x= landmark_near_x;
			double mu_y= landmark_near_y;

			// calculate normalization term
			double gauss_norm= (1.0/(2.0 * M_PI * sig_x * sig_y));

			// calculate exponent
			double exponent= ((x_obs - mu_x)*(x_obs - mu_x))/(2.0 * sig_x*sig_x) + ((y_obs - mu_y)*(y_obs - mu_y))/(2.0* sig_y*sig_y);
			// calculate weight using normalization terms and exponent
			double weight= gauss_norm *exp(-exponent);
			multiplier*=weight;

	}
	particles[j].weight = multiplier;
	weights[j] = particles[j].weight;

	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    vector<Particle> new_particles;

    random_device rd;
     mt19937 gen(rd());
     discrete_distribution<> distribution(weights.begin(), weights.end());

    for(int i = 0; i < num_particles; i++){
        Particle p = particles[distribution(gen)];
        new_particles.push_back(p);
    }
    particles = new_particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
