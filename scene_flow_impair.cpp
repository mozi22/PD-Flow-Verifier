/*****************************************************************************
**				Primal-Dual Scene Flow for RGB-D cameras					**
**				----------------------------------------					**
**																			**
**	Copyright(c) 2015, Mariano Jaimez Tarifa, University of Malaga			**
**	Copyright(c) 2015, Mohamed Souiai, Technical University of Munich		**
**	Copyright(c) 2015, MAPIR group, University of Malaga					**
**	Copyright(c) 2015, Computer Vision group, Tech. University of Munich	**
**																			**
**  This program is free software: you can redistribute it and/or modify	**
**  it under the terms of the GNU General Public License (version 3) as		**
**	published by the Free Software Foundation.								**
**																			**
**  This program is distributed in the hope that it will be useful, but		**
**	WITHOUT ANY WARRANTY; without even the implied warranty of				**
**  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the			**
**  GNU General Public License for more details.							**
**																			**
**  You should have received a copy of the GNU General Public License		**
**  along with this program.  If not, see <http://www.gnu.org/licenses/>.	**
**																			**
*****************************************************************************/

#include "scene_flow_impair.h"
#include "iostream"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <fstream>
#include <stdio.h>
#include <math.h>
using namespace std;

bool  fileExists(const std::string& path)
{
    return 0 == access(path.c_str(), 0x00 ); // 0x00 = Check for existence only!
}

PD_flow_opencv::PD_flow_opencv(unsigned int rows_config, 
	const char *intensity_filename_1, 
	const char *intensity_filename_2,
	const char *depth_filename_1,
	const char *depth_filename_2,
	const char* output_filename_root) {


	double m[9] = {570.342/2, 0, 320/2, 0, 570.342/2, 240/2, 0, 0, 1};
	camera_intrinsics = cv::Mat(3, 3, CV_64FC1);//, &m);

	int l =0;
	for(int i=0; i < 3; i++)
		for(int j=0; j < 3; j++,l++)
			camera_intrinsics.at<double>(i,j) = m[l];


	cout << "camera_intrinsic" << camera_intrinsics << endl;
	cout << "camera_intrinsic_inv" << camera_intrinsics.inv() << endl;


 //    rows = rows_config;      //Maximum size of the coarse-to-fine scheme - Default 240 (QVGA)
 //    cols = rows*320/240;
 //    ctf_levels = static_cast<unsigned int>(log2(float(rows/15))) + 1;
 //    fovh = M_PI*62.5f/180.f;
 //    fovv = M_PI*48.5f/180.f;

	// //Iterations of the primal-dual solver at each pyramid level.
	// //Maximum value set to 100 at the finest level
	// for (int i=5; i>=0; i--)
	// {
	// 	if (i >= ctf_levels - 1)
	// 		num_max_iter[i] = 100;	
	// 	else
	// 		num_max_iter[i] = num_max_iter[i+1]-15;
	// }

 //    //Compute gaussian mask
	// int v_mask[5] = {1,4,6,4,1};
 //    for (unsigned int i=0; i<5; i++)
 //        for (unsigned int j=0; j<5; j++)
 //            g_mask[i+5*j] = float(v_mask[i]*v_mask[j])/256.f;


 //    //Reserve memory for the scene flow estimate (the finest)
	// dxp = (float *) malloc(sizeof(float)*rows*cols);
	// dyp = (float *) malloc(sizeof(float)*rows*cols);
	// dzp = (float *) malloc(sizeof(float)*rows*cols);

    //Parameters of the variational method
    // lambda_i = 0.04f;
    // lambda_d = 0.35f;
    // mu = 75.f;

    // Set file names
    this->intensity_filename_1 = intensity_filename_1;
    this->intensity_filename_2 = intensity_filename_2;
    this->depth_filename_1 = depth_filename_1;
    this->depth_filename_2 = depth_filename_2;
    this->output_filename_root = output_filename_root;
}


void PD_flow_opencv::createImagePyramidGPU()
{
    //Copy new frames to the scene flow object
    csf_host.copyNewFrames(I, Z);

    //Copy scene flow object to device
    csf_device = ObjectToDevice(&csf_host);

    unsigned int pyr_levels = static_cast<unsigned int>(log2(float(width/cols))) + ctf_levels;
    GaussianPyramidBridge(csf_device, pyr_levels, cam_mode);

    //Copy scene flow object back to host
    BridgeBack(&csf_host, csf_device);
}

/*void PD_flow_opencv::solveSceneFlowGPU()
{
    unsigned int s;
    unsigned int cols_i, rows_i;
    unsigned int level_image;
    unsigned int num_iter;

    //For every level (coarse-to-fine)
    for (unsigned int i=0; i<ctf_levels; i++)
    {
        s = static_cast<unsigned int>(pow(2.f,int(ctf_levels-(i+1))));
        cols_i = cols/s;
        rows_i = rows/s;
        level_image = ctf_levels - i + static_cast<unsigned int>(log2(float(width/cols))) - 1;

        //=========================================================================
        //                              Cuda - Begin
        //=========================================================================

        //Cuda allocate memory
        csf_host.allocateMemoryNewLevel(rows_i, cols_i, i, level_image);

        //Cuda copy object to device
        csf_device = ObjectToDevice(&csf_host);

        //Assign zeros to the corresponding variables
        AssignZerosBridge(csf_device);

        //Upsample previous solution
        if (i>0)
            UpsampleBridge(csf_device);

        //Compute connectivity (Rij)
		RijBridge(csf_device);
		
		//Compute colour and depth derivatives
        ImageGradientsBridge(csf_device);
        WarpingBridge(csf_device);

        //Compute mu_uv and step sizes for the primal-dual algorithm
        MuAndStepSizesBridge(csf_detd::vector< cv::Mat > pixel3D;vice);

        //Primal-Dual solver
		for (num_iter = 0; num_iter < num_max_iter[i]; num_iter++)
        {
            GradientBridge(csf_device);
            DualVariablesBridge(csf_device);
            DivergenceBridge(csf_device);
            PrimalVariablesBridge(csf_device);
        }

        //Filter solution
        FilterBridge(csf_device);

        //Compute the motion field
        MotionFieldBridge(csf_device);

        //BridgeBack to host
        BridgeBack(&csf_host, csf_device);

        //Free memory of variables associated to this level
        csf_host.freeLevelVariables();

        //Copy motion field to CPU
		csf_host.copyMotionField(dxp, dyp, dzp);

		//For debugging
        //DebugBridge(csf_device);

        //=========================================================================
        //                              Cuda - end
        //=========================================================================
    }
}*/

// void PD_flow_opencv::freeGPUMemory()
// {
//     csf_host.freeDeviceMemory();
// }

void PD_flow_opencv::initializeCUDA()
{
	//Read one image to know the image resolution
	intensity1 = cv::imread(intensity_filename_1, CV_LOAD_IMAGE_GRAYSCALE);

	height = intensity1.rows;
	width = intensity1.cols;
	if (height == 240) {cam_mode = 2;}
	else			   {cam_mode = 1;}

	I = (float *) malloc(sizeof(float)*width*height);
	Z = (float *) malloc(sizeof(float)*width*height);   
	
	//Read parameters
    csf_host.readParameters(rows, cols, lambda_i, lambda_d, mu, g_mask, ctf_levels, cam_mode, fovh, fovv);

    //Allocate memory
    csf_host.allocateDevMemory();
}

// void PD_flow_opencv::showImages()
// {
// 	const unsigned int dispx = intensity1.cols + 20;
// 	const unsigned int dispy = intensity1.rows + 20;

// 	//Show images with OpenCV windows
// 	cv::namedWindow("I1", cv::WINDOW_AUTOSIZE);
// 	cv::moveWindow("I1",10,10);
// 	cv::imshow("I1", intensity1);

// 	cv::namedWindow("Z1", cv::WINDOW_AUTOSIZE);
// 	cv::moveWindow("Z1",dispx,10);
// 	cv::imshow("Z1", depth1);

// 	cv::namedWindow("I2", cv::WINDOW_AUTOSIZE);
// 	cv::moveWindow("I2",10,dispy);
// 	cv::imshow("I2", intensity2);

// 	cv::namedWindow("Z2", cv::WINDOW_AUTOSIZE);
// 	cv::moveWindow("Z2",dispx,dispy);
// 	cv::imshow("Z2", depth2);

// 	cv::waitKey(30);
// }

bool PD_flow_opencv::loadRGBDFrames()
{
	cv::Mat depth_float;


	//First intensity image
	intensity1 = cv::imread(intensity_filename_1, CV_LOAD_IMAGE_GRAYSCALE);
	if (intensity1.empty())
	{
		printf("\nThe first intensity image (%s) cannot be found, please check that it is in the correct folder \n", intensity_filename_1);
		return 0;
	}
	intensity1 = resizeImage(intensity1,cv::INTER_AREA);

	cv::imshow( "Display window0", intensity1 );

	// for (unsigned int u=0; u<width; u++)
	// 	for (unsigned int v=0; v<height; v++)
	// 		I[v + u*height] = float(intensity1.at<unsigned char>(v,u));

	//First depth image
	depth1 = cv::imread(depth_filename_1, -1);
	if (depth1.empty())
	{
		printf("\nThe first depth image (%s) cannot be found, please check that it is in the correct folder \n", depth_filename_1);
		return 0;
	}

	depth1.convertTo(depth_float, CV_32FC1, 1.0 / 5000.0);
	depth1 = depth_float;
	depth1 = resizeImage(depth1,cv::INTER_NEAREST);


	// cout<< "depth image cols = " << depth1.cols << endl;
	// cout<< "depth image rows = " << depth1.rows << endl;

	// for (unsigned int v=0; v<height; v++)
	// 	for (unsigned int u=0; u<width; u++)
	// 		Z[v + u*height] = depth_float.at<float>(v,u);

	// createImagePyramidGPU();

	std::vector< cv::Mat > pixel3D = projectImagePxTo3D(intensity1,depth1);

	// // now we add sceneflow from file
	pixel3D = apply_optical_flow_to_img_1(pixel3D);

	// // project the matrix back to 2D.
	pixel3D = projectBackTo2D(pixel3D);

	// writeTxt(pixel3D);


	//Second intensity image
	intensity2 = cv::imread(intensity_filename_2, CV_LOAD_IMAGE_GRAYSCALE);
	if (intensity2.empty())
	{
		printf("\nThe second intensity image (%s) cannot be found, please check that it is in the correct folder \n", intensity_filename_2);
		return 0;
	}

	intensity2 = resizeImage(intensity2,cv::INTER_AREA);
	interpolatePixelRGB(pixel3D,intensity2);
	// intensity2 = resizeImage(intensity2);
	// for (unsigned int v=0; v<height; v++)
	// 	for (unsigned int u=0; u<width; u++)
	// 		I[v + u*height] = float(intensity2.at<unsigned char>(v,u));

	// interpolatePixelRGB(pixel3D,intensity2);

	//Second depth image
	depth2 = cv::imread(depth_filename_2, -1);
	if (depth2.empty())
	{
		printf("\nThe second depth image (%s) cannot be found, please check that they are in the correct folder \n", depth_filename_2);
		return 0;
	}
	depth2.convertTo(depth_float, CV_32FC1, 1.0 / 5000.0);
	// for (unsigned int v=0; v<height; v++)
	// 	for (unsigned int u=0; u<width; u++)
	// 		Z[v + u*height] = depth_float.at<float>(v,u);

	// createImagePyramidGPU();
	// cv::Mat img23D = projectImagePxTo3D(intensity2);

	cout << "hogya" << endl;
	return 1;
}

// Create the image
// cv::Mat PD_flow_opencv::createImage() const
// {
// 	//Save scene flow as an RGB image (one colour per direction)
// 	cv::Mat sf_image(rows, cols, CV_8UC3);

//     //Compute the max values of the flow (of its components)
// 	float maxmodx = 0.f, maxmody = 0.f, maxmodz = 0.f;
// 	for (unsigned int v=0; v<rows; v++)
// 		for (unsigned int u=0; u<cols; u++)
// 		{
//             if (fabs(dxp[v + u*rows]) > maxmodx)
//                 maxmodx = fabs(dxp[v + u*rows]);
//             if (fabs(dyp[v + u*rows]) > maxmody)
//                 maxmody = fabs(dyp[v + u*rows]);
//             if (fabs(dzp[v + u*rows]) > maxmodz)
//                 maxmodz = fabs(dzp[v + u*rows]);
// 		}

// 	//Create an RGB representation of the scene flow estimate: 
// 	for (unsigned int v=0; v<rows; v++)
// 		for (unsigned int u=0; u<cols; u++)
// 		{
//             sf_image.at<cv::Vec3b>(v,u)[0] = static_cast<unsigned char>(255.f*fabs(dxp[v + u*rows])/maxmodx); //Blue - x
//             sf_image.at<cv::Vec3b>(v,u)[1] = static_cast<unsigned char>(255.f*fabs(dyp[v + u*rows])/maxmody); //Green - y
//             sf_image.at<cv::Vec3b>(v,u)[2] = static_cast<unsigned char>(255.f*fabs(dzp[v + u*rows])/maxmodz); //Red - z
// 		}
	
// 	return sf_image;
// }

/**
 * Save results without displaying them
 */
// void PD_flow_opencv::saveResults( const cv::Mat& sf_image ) const
// {
// 	//Save the scene flow as a text file 
// 	char	name[500];
// 	int     nFichero = 0;
// 	bool    free_name = false;

// 	while (!free_name)
// 	{
// 		nFichero++;
// 		sprintf(name, "%s_results%02u.txt", output_filename_root, nFichero );
// 		free_name = !fileExists(name);
// 	}
	
// 	std::ofstream f_res;
// 	f_res.open(name);
// 	printf("Saving the estimated scene flow to file: %s \n", name);

// 	//Format: (pixel(row), pixel(col), vx, vy, vz)
// 	for (unsigned int v=0; v<rows; v++)
// 		for (unsigned int u=0; u<cols; u++)
// 		{
// 			f_res << v << " ";
// 			f_res << u << " ";
// 			f_res << dxp[v + u*rows] << " ";
// 			f_res << dyp[v + u*rows] << " ";
// 			f_res << dzp[v + u*rows] << std::endl;
// 		}

// 	f_res.close();

// 	//Save the RGB representation of the scene flow
// 	sprintf(name, "%s_representation%02u.png", output_filename_root, nFichero);
// 	printf("Saving the visual representation to file: %s \n", name);
// 	cv::imwrite(name, sf_image);
// }


// void PD_flow_opencv::showAndSaveResults( )
// {
// 	cv::Mat sf_image = createImage( );

// 	//Show the scene flow as an RGB image	
// 	cv::namedWindow("SceneFlow", cv::WINDOW_NORMAL);
//     cv::moveWindow("SceneFlow",width - cols/2,height - rows/2);
// 	cv::imshow("SceneFlow", sf_image);
// 	cv::waitKey(100000);

// 	saveResults( sf_image );
// }







// step 4
void PD_flow_opencv::writeTxt(std::vector< cv::Mat > finalPixel2D){

	cout << "item count is " << finalPixel2D.size() << endl;
	std::ofstream f_res;
	f_res.open("intermediate_results.txt");

	int index_x = 0;
	int index_y = 0;

	//Format: (pixel(row), pixel(col), vx, vy, vz)
	for (unsigned int v=0; v<finalPixel2D.size(); v++)
	{
		index_y = v / 320;
		index_x = v- index_y*320;
		cv::Mat pixel2D = finalPixel2D[v];

		f_res << index_y << " ";
		f_res << index_x << " ";
		f_res << pixel2D.at<double>(0,0) << " ";
		f_res << pixel2D.at<double>(1,0) << " ";
		f_res << pixel2D.at<double>(2,0) << std::endl;

	}

	f_res.close();
}


std::vector< cv::Mat > PD_flow_opencv::apply_optical_flow_to_img_1(std::vector< cv::Mat > pixel3D){

	int posX, posY;
	double u,v,w;
	int i = 0;

	std::ifstream infile("./pdflow_results02.txt");
	while (infile >> posX >> posY >> u >> v >> w)
	{
		cv::Mat item = pixel3D[i];


		if(item.at<double>(0,0) != 0 && item.at<double>(1,0) != 0 && item.at<double>(2,0) != 0){
			// cout << "before sceneflow = matrix = " << pixel3D[i] << endl;
			// cout << "depth before = " << item.at<double>(2,0) << endl;
			// cout << "sceneflowX before = " << sceneflowX <<endl;
			// cout << "sceneflowY before = " << sceneflowY <<endl;
			// cout << "sceneflowZ before = " << sceneflowZ <<endl;
	
		}

		item.at<double>(0,0) = item.at<double>(0,0) + u;
		item.at<double>(1,0) = item.at<double>(1,0) + v;
		// item.at<double>(2,0) = item.at<double>(2,0) + sceneflowZ;



		pixel3D[i] = item;

		if(item.at<double>(0,0) != 0){
			// cout << "sceneflowX = " << sceneflowX <<endl;
			// cout << "sceneflowY = " << sceneflowY <<endl;
			// cout << "sceneflowZ = " << sceneflowZ <<endl;
	
			// cout << "depth after = " << item.at<double>(2,0) << endl;

			// cout << "after sceneflow = matrix = " << pixel3D[i] << endl;
		}
		i++;
	}

	return pixel3D;

}




double PD_flow_opencv::bilinearInterpolation(double q11,double q12,double q21,double q22,
							double x1,double x2,double y1,double y2,double x, double y){

	// source: http://supercomputingblog.com/graphics/coding-bilinear-interpolation/

	double r1 = ((x2-x)/(x2-x1))*q11 + ((x-x1)/(x2-x1))* q21;
	double r2 = ((x2-x)/(x2-x1))*q12 + ((x-x1)/(x2-x1))* q22;

	return ((y2 - y)/(y2-y1))*r1 + ((y-y1)/(y2-y1))*r2;
}

// step 4
double PD_flow_opencv::interpolatePixelRGB(std::vector< cv::Mat > finalPixel2D,cv::Mat intensity2){

	cv::Mat finalImage(240,320, CV_8UC3, cv::Scalar(0,0,255));
	int row = 0;
	int col = 0;

	for(int row = 0; row < finalImage.rows; ++row)
		for(int col = 0; col < finalImage.cols; ++col)
		{
			int i = row*finalImage.cols + col;

	
		cv::Mat pixel2D = finalPixel2D[i];

		// if both the pixel values are 0 or one of them is nan. just move to the next one.
		if((pixel2D.at<double>(0,0) == 0 && pixel2D.at<double>(1,0) == 0) ||
			isnan(pixel2D.at<double>(0,0)) ==  1 || isnan(pixel2D.at<double>(1,0)) ==  1){
			finalImage.at<cv::Vec3b>(row,col) = cv::Vec3b(0,255,0);
			//continue;
		}

		/* select top left pixel rgb Q12 */
		int Q12_X = floor(pixel2D.at<double>(0,0));
		int Q12_Y = floor(pixel2D.at<double>(1,0));

		/* select top right pixel rgb Q22 */
		int Q22_X = ceil(pixel2D.at<double>(0,0));
		int Q22_Y = floor(pixel2D.at<double>(1,0));

		/* select bottom left pixel rgb Q11 */
		int Q11_X = floor(pixel2D.at<double>(0,0));
		int Q11_Y = ceil(pixel2D.at<double>(1,0));

		/* select bottom right pixel rgb Q21 */
		int Q21_X = ceil(pixel2D.at<double>(0,0));
		int Q21_Y = ceil(pixel2D.at<double>(1,0));

		if(Q12_X >= 0 && Q21_X < 320 && Q12_Y >= 0 && Q21_Y < 240){}
		else{
			finalImage.at<cv::Vec3b>(row,col) = cv::Vec3b(255,0,0); //blue
			continue;
			
		}

		cv::Vec3f Q12RGB = intensity2.at<cv::Vec3f>(Q12_X,Q12_Y);
		cv::Vec3f Q22RGB = intensity2.at<cv::Vec3f>(Q22_X,Q22_Y);
		cv::Vec3f Q11RGB = intensity2.at<cv::Vec3f>(Q11_X,Q11_Y);
		cv::Vec3f Q21RGB = intensity2.at<cv::Vec3f>(Q21_X,Q21_Y);
		cv::Vec3f tmp = intensity2.at<cv::Vec3b>(Q21_X,Q21_Y);
		unsigned char tmp2 = intensity2.at<unsigned char>(Q21_Y,Q21_X);

		// if(row == 2 && col == 207){
		// 	cout << "R values Q11 = " << Q11RGB[0] << endl;
		// 	cout << "R values Q12 = " << Q12RGB[0] << endl;
		// 	cout << "R values Q22 = " << Q22RGB[0] << endl;
		// 	cout << "R values Q21 = " << Q21RGB[0] << endl;
		// }

		// applying interpolation for RGB
		// double r_value = bilinearInterpolation(Q11RGB[0],Q12RGB[0],Q21RGB[0],Q22RGB[0],Q11_X,Q22_X,Q11_Y,Q22_Y,pixel2D.at<double>(0,0),pixel2D.at<double>(1,0));
		// double g_value = bilinearInterpolation(Q11RGB[1],Q12RGB[1],Q21RGB[1],Q22RGB[1],Q11_X,Q22_X,Q11_Y,Q22_Y,pixel2D.at<double>(0,0),pixel2D.at<double>(1,0));
		// double b_value = bilinearInterpolation(Q11RGB[2],Q12RGB[2],Q21RGB[2],Q22RGB[2],Q11_X,Q22_X,Q11_Y,Q22_Y,pixel2D.at<double>(0,0),pixel2D.at<double>(1,0));
		cout << row << "  " << col << "\n";
		cv::Vec3b color;// = finalImage.at<cv::Vec3b>(row,col);
		color[0] = Q21RGB[0];
		color[1] = Q21RGB[1];
		color[2] = Q21RGB[2];

		
		color[0] = tmp[0];
		color[1] = tmp[0];
		color[2] = tmp[0];
		color[0] = tmp2;
		color[1] = tmp2;
		color[2] = tmp2;

		finalImage.at<cv::Vec3b>(row,col) = color;


		// col++;

		// if(col == 240){
		// 	row++;
		// 	col = 0;		
		// }

	}
		cout << "printing image" << endl;

	cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
	cv::imshow( "Display window", finalImage );
	cv::imshow( "Display window2", intensity2 );
	cv::waitKey(0);
}

// step 2
std::vector< cv::Mat > PD_flow_opencv::applySceneFlowTo3DWorldPixels(std::vector< cv::Mat > pixel3D){

	int posX, posY;
	double sceneflowX, sceneflowY, sceneflowZ;
	int i = 0;

	std::ifstream infile("./pdflow_results02.txt");
	while (infile >> posX >> posY >> sceneflowX >> sceneflowY >> sceneflowZ)
	{
		cv::Mat item = pixel3D[i];


		if(item.at<double>(0,0) != 0 && item.at<double>(1,0) != 0 && item.at<double>(2,0) != 0){
			// cout << "before sceneflow = matrix = " << pixel3D[i] << endl;
			// cout << "depth before = " << item.at<double>(2,0) << endl;
			// cout << "sceneflowX before = " << sceneflowX <<endl;
			// cout << "sceneflowY before = " << sceneflowY <<endl;
			// cout << "sceneflowZ before = " << sceneflowZ <<endl;
		}

		item.at<double>(0,0) = item.at<double>(0,0) + sceneflowX;
		item.at<double>(1,0) = item.at<double>(1,0) + sceneflowY;
		item.at<double>(2,0) = item.at<double>(2,0) + sceneflowZ;

		pixel3D[i] = item;

		if(item.at<double>(0,0) != 0){
			// cout << "sceneflowX = " << sceneflowX <<endl;
			// cout << "sceneflowY = " << sceneflowY <<endl;
			// cout << "sceneflowZ = " << sceneflowZ <<endl;
	
			// cout << "depth after = " << item.at<double>(2,0) << endl;

			// cout << "after sceneflow = matrix = " << pixel3D[i] << endl;
		}
		i++;
	}

	return pixel3D;
}

// step 3
std::vector< cv::Mat > PD_flow_opencv::projectBackTo2D(std::vector< cv::Mat > pixelWorld3D){

	for(int i=0;i<pixelWorld3D.size(); i++){

		// going back to 2D plane in homogenous coordinates
		cv::Mat item = camera_intrinsics * pixelWorld3D[i];

		// going back to cartesian coordinates
		item.at<double>(0,0) = item.at<double>(0,0) / item.at<double>(2,0);
		item.at<double>(1,0) = item.at<double>(1,0) / item.at<double>(2,0);
		item.at<double>(2,0) = item.at<double>(2,0) / item.at<double>(2,0);

		pixelWorld3D[i] = item;
	}

	return pixelWorld3D;
}



cv::Mat PD_flow_opencv::resizeImage(cv::Mat image, int interpolation){

	cv::Mat dst;

	cv::resize(image, dst, cv::Size(320,240),0,0,interpolation);

	return dst;
}

// step 1
std::vector< cv::Mat > PD_flow_opencv::projectImagePxTo3D(cv::Mat image, cv::Mat depth_image)
{
	cout << "inv in function " << this->camera_intrinsics.inv() << endl;

	cv::Mat inv_cam_intr = camera_intrinsics.inv();
	std::vector< cv::Mat > pixel3D;

	for(int i=0;i<image.rows;i++){

		for(int j=0;j<image.cols;j++){

			// double m[3][1] = {{(double)i}, {(double)j}, {1.00}};
			// cv::Mat pixel2D = cv::Mat(3, 1, CV_32F, m).inv();
			// cv::Mat pixel2D = (cv::Mat_<double>(3,1) << (double)i, (double)j, 1.00);

			double pixel2DArr[3] = { (double)j, (double)i, 1.00 };

			cv::Mat pixel2D = cv::Mat(3,1, CV_64FC1);

			for(int i=0; i < 3; i++)
				pixel2D.at<double>(i,0) = pixel2DArr[i];

			/* 
			*  Multiply the pixel position matrix [x,y,1] with camera_intrinsic matrix.
			*  This gives us the projection of pixel on the image the image plane w.r.t the camera.
			*/

			// cout << "rows = " << camera_intrinsics.inv().rows << endl;
			// cout << "cols = " << camera_intrinsics.inv().cols << endl;

			// cout << "correct one "<<endl;
			// printMatrix(camera_intrinsics);
			cv::Mat resulting2Dpixel(3,1,CV_64FC1);
			// resulting2Dpixel = camera_intrinsics.inv() * pixel2D;

		 	resulting2Dpixel = inv_cam_intr * pixel2D;

			// cout << " index i,j = "<< i << "," << j<< " ======= depth value =  " << depth_image.at<double>(i,j) << " ====== sum = " <<  resulting2Dpixel * depth_image.at<double>(i,j) <<endl;

			// printMatrix(resulting2Dpixel);
			// cout << "endbreak" << endl;
			// printMatrix(resulting2Dpixel * depth_image.at<double>(i,j) );
			/* Adding the depth in the image. pixel 3D should be [x,y,z] i.e 3x1 matrix */

		 	// cout << " resulting 2d x = " << resulting2Dpixel.at<double>(0,0) << endl;
		 	// cout << " resulting 2d y = " << resulting2Dpixel.at<double>(1,0) << endl;
		 	// cout << " resulting 2d z = " << resulting2Dpixel.at<double>(2,0) << endl;
		 	// cout << " depth pixel " << depth_image.at<double>(i,j) << endl;

		 	// cout << "depth intensity = "  << depth_image.at<float>(i,j) << endl;

			// pixel3D.push_back(resulting2Dpixel * depth_image.at<float>(i,j));
			pixel3D.push_back(resulting2Dpixel);

			// cv::Mat matrix_with_depth = resulting2Dpixel * depth_image.at<double>(i,j);

			// // going back to 2D plane in homogenous coordinates
			// cv::Mat item = camera_intrinsics * matrix_with_depth;

			// // going back to cartesian coordinates
			// item.at<double>(0,0) = item.at<double>(0,0) / item.at<double>(2,0);
			// item.at<double>(1,0) = item.at<double>(1,0) / item.at<double>(2,0);
			// item.at<double>(2,0) = item.at<double>(2,0) / item.at<double>(2,0);

		 // 	cout << "resulting 2d pixel" << resulting2Dpixel << "  " << j << " "  << i << endl;
			// cout << "final matrix = " << item << endl;

		}
	}

	cout << " vector size = " << pixel3D.size() << endl;
	return pixel3D;
}


void PD_flow_opencv::printMatrix(cv::Mat image){

	for(int i=0;i<image.rows;i++){

		for(int j=0;j<image.cols;j++){

			cout << "index(" << i << "," << j <<") = " << image.at<double>(i,j) << endl;
		}
	}
}