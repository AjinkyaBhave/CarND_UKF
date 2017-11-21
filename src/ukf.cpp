#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;
  
  // Noise vector dimension
  n_noise_ = 2;

  // Sigma point spreading parameter
  double lambda_ = 3;
  
  // initial state vector
  x_ = VectorXd(n_x_);
  x_ << 0,0,0,0,0;

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);
  P_ <<  1,0,0,0,0,
			0,1,0,0,0,
			0,0,1,0,0,
			0,0,0,1,0,
			0,0,0,0,1;
			
	// Predicted sigma points matrix
	Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
	
	// Process noise matrix
	P_noise_ << 0, std_a_*std_a_, 0, std_yawdd_*std_yawdd_;
	
	// Weights of sigma points
	weights_ = VectorXd(2 * n_aug_ + 1);
	weights_(0) = lambda_/(lambda_ + n_aug_);
	for (int i=1;i<(2*n_aug_+1);i++){
		weights_(i) = 0.5/(lambda_+n_aug_);
	}
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} measurement_pack The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage measurement_pack) {
	// Initialisation
	if (!is_initialized_) {
		// first measurement
		cout << "First measurement.\n ";

		if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
		// Convert radar from polar to cartesian coordinates and initialize state.
		x_(0) = measurement_pack.raw_measurements_[0]* cos(measurement_pack.raw_measurements_[1]);
		x_(1) = measurement_pack.raw_measurements_[0]* sin(measurement_pack.raw_measurements_[1]);
		// Cannot calculate velocity from range rate directly, so assuming zero.
		x_(2) = 0;
		// Assuming bicyclist driving straight with zero turning rate.
		x_(3) = 0;
		x_(4) = 0;
		}
		else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
		//set the state with the initial location, zero velocity, and driving straight.
		x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0, 0;
		}
		else{
		 cout << "Invalid sensor type.\n";
		 return;
		}
	// Update time stamp for prediction step
	previous_timestamp_ = measurement_pack.timestamp_;
	// done initializing, no need to predict or update
	is_initialized_ = true;
	return;
	}
	 
	// Prediction
	//compute the time elapsed between the current and previous measurements in seconds
	float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
	previous_timestamp_ = measurement_pack.timestamp_;
	 
	// Run prediction step
	Prediction(dt);
	
	// Update
	if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
	// Radar updates
		UpdateRadar(measurement_pack);
	} 
	else {
   // Laser updates
		UpdateLidar(measurement_pack);
	}

	// print the output
	cout << "x_ = " << x_ << endl;
	cout << "P_ = " << P_ << endl;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  
  // Create sigma points
  // create augmented mean state
	VectorXd x_aug = VectorXd(n_aug_);
	x_aug.head(n_x_) = x_;
	x_aug(n_aug_-2) = 0;
	x_aug(n_aug_-1) = 0;
	// create augmented covariance matrix
	MatrixXd P_aug = MatrixXd(n_aug_,n_aug_);
	P_aug.topLeftCorner(n_x_,n_x_) = P_;
	P_aug.bottomRightCorner(n_noise_, n_noise_) = P_noise_;
	//create square root matrix
	MatrixXd A = P_aug.llt().matrixL();
	A  *= sqrt(lambda_+n_aug_);
	// create augmented sigma points matrix
	MatrixXd Xsig_aug = MatrixXd(n_aug_,2*n_aug_+1);
	Xsig_aug.col(0) = x_aug;
	for (int i=0;i<n_aug_;i++){
		Xsig_aug.col(i+1) = x_aug + A.col(i);
		Xsig_aug.col(i+1+n_aug_) = x_aug - A.col(i);
	}
	
	// Predict sigma points
	VectorXd x 		= VectorXd(n_x_);
	VectorXd x_dot	= VectorXd(n_x_);
	VectorXd nu 	= VectorXd(n_noise_);
	VectorXd nu_dot = VectorXd(n_x_);
	//predict 2*(n_aug_)+1 sigma points
	for (int i=0; i < (2*n_aug_+1);i++)
	{
	 x_aug = Xsig_aug.col(i);
	 x = x_aug.head(n_x_);
	 nu = x_aug.tail(n_noise_);
	 
	 // Create state derivative vector
	 // Avoid division by zero
	 if (fabs(x(4)) > 0.001){
		x_dot(0) = (x(2)/x(4))*(sin(x(3)+x(4)*delta_t) - sin(x(3)) );
		x_dot(1) = (x(2)/x(4))*(-cos(x(3)+x(4)*delta_t) + cos(x(3)) );
		x_dot(3) = x(4)*delta_t;
	 }
	 else{
		x_dot(0) = x(2)*cos(x(3)*delta_t);
		x_dot(1) = x(2)*sin(x(3)*delta_t);
		x_dot(3) = 0;
	 }
	 x_dot(2) = 0;
	 x_dot(4) = 0;
	 
	 // Create noise vector
	 nu_dot(0) = 0.5*(delta_t*delta_t)*cos(x(3))*nu(0);
	 nu_dot(1) = 0.5*(delta_t*delta_t)*sin(x(3))*nu(0);
	 nu_dot(2) = delta_t*nu(0);
	 nu_dot(3) = 0.5*(delta_t*delta_t)*nu(1);
	 nu_dot(4) = delta_t*nu(1);
	// Write predicted sigma points into correct column
	 Xsig_pred_.col(i) = x + x_dot + nu_dot;
	}
	
	// Predict state and covariance
	// Calculate mean
  x_.fill(0);
  for (int i=0;i<(2*n_aug_+1);i++){
		x_ += weights_(i)*Xsig_pred_.col(i);
  }
  // Calculate covariance
  P_.fill(0);
  // state difference
  VectorXd x_diff = VectorXd(n_x_);
  for (int i=0;i<(2*n_aug_+1);i++){
	 x_diff = Xsig_pred_.col(i) - x_;
	 // Angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
	 P_ += weights_(i)*x_diff*x_diff.transpose();
  }  
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} measurement_pack
 */
void UKF::UpdateLidar(MeasurementPackage measurement_pack) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} measurement_pack
 */
void UKF::UpdateRadar(MeasurementPackage measurement_pack) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
}
