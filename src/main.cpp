#include <stdio.h>
#include <FeatureMatch.h>

FeatureMatch *featureMatch;

int main(int argc, char** argv )
{
	featureMatch = new FeatureMatch();
	while(1){
		featureMatch->Matching();
	}
	return 1;
}
