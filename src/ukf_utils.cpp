  //your code goes here 
void GenerateSigmaPoints(&Xsig_in)
{
	//calculate sigma points ...
	//set sigma points as columns of matrix Xsig
	int n_noise = 2;
	MatrixXd P_noise = MatrixXd(n_noise,n_noise);
	P_noise << 0, std_a*std_a, 0, std_yawdd*std_yawdd;

	//create augmented mean state
	x_aug.head(n_x) = x;
	x_aug(n_aug-2) = 0;
	x_aug(n_aug-1) = 0;
	//create augmented covariance matrix
	P_aug.topLeftCorner(n_x,n_x) = P;
	P_aug.bottomRightCorner(n_noise, n_noise) = P_noise;
	//create square root matrix
	MatrixXd A = P_aug.llt().matrixL();
	//create augmented sigma points
	A  *= sqrt(lambda+n_aug);
	Xsig_aug.col(0) = x_aug;
	for (int i=0;i<n_aug;i++){
		Xsig_aug.col(i+1) = x_aug + A.col(i);
		Xsig_aug.col(i+1+n_aug) = x_aug - A.col(i);
	}
}
  
void PredictSigmaPoints(&XSig_Pred_in)
{
  int n_noise = 2;
  VectorXd x 		= VectorXd(n_x);
  VectorXd x_dot	= VectorXd(n_x);
  VectorXd nu 		= VectorXd(n_noise);
  VectorXd nu_dot = VectorXd(n_x);
  VectorXd x_aug  = VectorXd(n_aug);
  
  //predict sigma points
  for (int i=0; i < (2*n_aug+1);i++)
  {
	 x_aug = Xsig_aug.col(i);
	 x = x_aug.head(n_x);
	 nu = x_aug.tail(n_noise);
	 
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
	// Write predicted sigma points into right column
	 Xsig_pred.col(i) = x + x_dot + nu_dot;
	 /*Xsig_pred(0,i) = 
	 Xsig_pred(1,i) = 
	 Xsig_pred(2,i) = 
	 Xsig_pred(3,i) =
	 Xsig_pred(4,i) = 
	*/ 
  }
}

void PredictMeanAndCovariance(){
	//create vector for weights
  VectorXd weights = VectorXd(2*n_aug+1);
  
  //create vector for predicted state
  VectorXd x = VectorXd(n_x);

  //create covariance matrix for prediction
  MatrixXd P = MatrixXd(n_x, n_x);
  
  // Set weights
  weights(0) = lambda/(lambda+n_aug);
  for (int i=1;i<(2*n_aug+1);i++){
      weights(i) = 0.5/(lambda+n_aug);
  }
  // Calculate mean
  x.fill(0);
  for (int i=0;i<(2*n_aug+1);i++){
		x += weights(i)*Xsig_pred.col(i);
  
  }
  // Calculate covariance
  P.fill(0);
  // state difference
  VectorXd x_diff = VectorXd(n_x);
  for (int i=0;i<(2*n_aug+1);i++){
	 x_diff = Xsig_pred.col(i) - x;
	 // Angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
	 P += weights(i)*x_diff*x_diff.transpose();
  }  
}

void PredictRadarMeasurement()
{
  VectorXd x_sig = VectorXd(n_x);
  VectorXd z_sig = VectorXd(n_z);
  MatrixXd R_radar = MatrixXd(n_z,n_z);
  R_radar << std_radr*std_radr, 0, 0, 
             0, std_radphi*std_radphi, 0,
             0, 0, std_radrd*std_radrd;
  //transform sigma points into measurement space
  for (int i=0; i < (2*n_aug+1);i++)
  {
	 x_sig = Xsig_pred.col(i);
	 z_sig(0) = sqrt(x_sig(0)*x_sig(0) + x_sig(1)*x_sig(1));
	 z_sig(1) = atan2(x_sig(1), x_sig(0));
	 z_sig(2) = (x_sig(0)*cos(x_sig(3))*x_sig(2)+x_sig(1)*sin(x_sig(3))*x_sig(2))/z_sig(0);
	 Zsig.col(i) = z_sig;
  }
  //calculate mean predicted measurement
  z_pred.fill(0);
  for (int i=0;i<(2*n_aug+1);i++){
		z_pred += weights(i)*Zsig.col(i);
  }
  //calculate measurement covariance matrix S
  S.fill(0);
  // measurement difference
  VectorXd z_diff = VectorXd(n_z);
  for (int i=0;i<(2*n_aug+1);i++){
	 z_diff = Zsig.col(i) - z_pred;
	 // Angle normalization
     while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
     while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
	 S += weights(i)*z_diff*z_diff.transpose();
  }
  S += R_radar;
}

void UpdateState()
{
  //calculate cross correlation matrix
  //calculate measurement covariance matrix S
  Tc.fill(0);
  // state difference
  VectorXd x_diff = VectorXd(n_x);
  // measurement difference
  VectorXd z_diff = VectorXd(n_z);
  for (int i=0;i<(2*n_aug+1);i++){
	 x_diff = Xsig_pred.col(i) - x;
	 // Angle normalization
     while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
     while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
	 z_diff = Zsig.col(i) - z_pred;
	 // Angle normalization
     while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
     while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
	 Tc += weights(i)*x_diff*z_diff.transpose();
  }
  //calculate Kalman gain K;
  MatrixXd K  = Tc * S.inverse();
  //update state mean and covariance matrix
  x = x + K*(z-z_pred);
  P = P - K*S*K.transpose();
}