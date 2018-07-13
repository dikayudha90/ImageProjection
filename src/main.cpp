#include <stdio.h>
//#include <FeatureMatch.h>
//#include <Projection.h>
#include "VisualOdometry.h"
//Projection *project;
//FeatureMatch *ok;

PoseCalculator *visual;

int main(int argc, char** argv )
{
	//ok = new FeatureMatch();
	//project = new Projection();
	visual = new PoseCalculator();
	//while(1){
	visual->CalculatePose();
	//featureMatch->RANSACKNNMatch();
	//ok->KAZERANSAC();
	//project->Project();
	//}
	return 1;
}
