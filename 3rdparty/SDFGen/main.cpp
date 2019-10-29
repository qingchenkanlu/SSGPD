//SDFGen - A simple grid-based signed distance field (level set) generator for triangle meshes.
//Written by Christopher Batty (christopherbatty@yahoo.com, www.cs.columbia.edu/~batty)
//...primarily using code from Robert Bridson's website (www.cs.ubc.ca/~rbridson)
//This code is public domain. Feel free to mess with it, let me know if you like it.

#include "makelevelset3.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <limits>

int main(int argc, char* argv[]) {
  
  if(argc != 4) {
    std::cout << "SDFGen - A utility for converting closed oriented triangle meshes into grid-based signed distance fields.\n";
    std::cout << "\nThe output file format is:";
    std::cout << "<ni> <nj> <nk>\n";
    std::cout << "<origin_x> <origin_y> <origin_z>\n";
    std::cout << "<dx>\n";
    std::cout << "<value_1> <value_2> <value_3> [...]\n\n";
    
    std::cout << "(ni,nj,nk) are the integer dimensions of the resulting distance field.\n";
    std::cout << "(origin_x,origin_y,origin_z) is the 3D position of the grid origin.\n";
    std::cout << "<dx> is the grid spacing.\n\n";
    std::cout << "<value_n> are the signed distance data values, in ascending order of i, then j, then k.\n";

    std::cout << "The output filename will match that of the input, with the OBJ suffix replaced with SDF.\n\n";

    std::cout << "Usage: SDFGen <filename> <dim> <padding>\n\n";
    std::cout << "Where:\n";
    std::cout << "\t<filename> specifies a Wavefront OBJ (text) file representing a *triangle* mesh (no quad or poly meshes allowed). File must use the suffix \".obj\".\n";
    //    std::cout << "\t<dx> specifies the length of grid cell in the resulting distance field.\n";
    std::cout << "\t<dim> specifies the dimension of the cube\n\n";
    std::cout << "\t<padding> specifies the number of cells worth of padding between the object bound box and the boundary of the distance field grid. Minimum is 1.\n\n";
    //    std::cout << "\t<outdir> the directory to write the output to\n\n";
    
    exit(-1);
  }

  std::string filename(argv[1]);
  if(filename.size() < 5 || filename.substr(filename.size()-4) != std::string(".obj")) {
    std::cerr << "Error: Expected OBJ file with filename of the form <name>.obj.\n";
    exit(-1);
  }

  std::stringstream arg2(argv[2]);
  float dx;
  int dim;
  //  arg2 >> dx;
  arg2 >> dim;

  std::stringstream arg3(argv[3]);
  int padding;
  arg3 >> padding;

  if(padding < 1) padding = 1;
  if(2*padding > dim) {
    std::cout << "Padding greater than cube dim" << std::endl;
    exit(1);
  }

  //start with a massive inside out bound box.
  Vec3f min_box(std::numeric_limits<float>::max(),std::numeric_limits<float>::max(),std::numeric_limits<float>::max()), 
    max_box(-std::numeric_limits<float>::max(),-std::numeric_limits<float>::max(),-std::numeric_limits<float>::max());
  
  std::cout << "Reading data.\n";

  std::ifstream infile(argv[1]);
  if(!infile) {
    std::cerr << "Failed to open. Terminating.\n";
    exit(-1);
  }

  int ignored_lines = 0;
  int count = 0;
  std::string slash = "/";
  std::string line;
  std::vector<Vec3f> vertList;
  std::vector<Vec3ui> faceList;
  while(!infile.eof()) {
    std::getline(infile, line);
    if(line.substr(0,1) == std::string("v")) {
      std::stringstream data(line);
      char c;
      Vec3f point;
      data >> c >> point[0] >> point[1] >> point[2];
      vertList.push_back(point);
      update_minmax(point, min_box, max_box);
    }
    else if(line.substr(0,1) == std::string("f")) {
      std::stringstream data(line);
      char c;
      std::string v0str, v1str, v2str;
      int v0,v1,v2;
      data >> c >> v0str >> v1str >> v2str;
      
      // get vertex index from slash parsing
      size_t pos = 0;
      std::string token = v0str;
      if ((pos = v0str.find(slash)) != std::string::npos) {
        token = v0str.substr(0, pos);
      }
      v0 = atoi(token.c_str());

      pos = 0;
      token = v1str;
      if ((pos = v1str.find(slash)) != std::string::npos) {
        token = v1str.substr(0, pos);
      }
      v1 = atoi(token.c_str());

      pos = 0;
      token = v2str;
      if ((pos = v2str.find(slash)) != std::string::npos) {
        token = v2str.substr(0, pos);
      }
      v2 = atoi(token.c_str());

      // if (v0 - 1 == 2649) {
      //   std::cout << "line " << line << std::endl;
      //   std::cout << "data " << c << " " << v0 << " " << v1 << " " << v2 << std::endl;
      // }
      count++;
      faceList.push_back(Vec3ui(v0-1,v1-1,v2-1));
    }
    else {
      ++ignored_lines; 
    }
  }
  infile.close();
  
  if(ignored_lines > 0)
    std::cout << "Warning: " << ignored_lines << " lines were ignored since they did not contain faces or vertices.\n";

  std::cout << "Read in " << vertList.size() << " vertices and " << faceList.size() << " faces." << std::endl;

  //Add padding around the box.
  Vec3f unit(1,1,1);
  // min_box -= padding*dx*unit;
  // max_box += padding*dx*unit;
  float max_diff = std::max(max_box.v[0] - min_box.v[0], std::max(max_box.v[1] - min_box.v[1], max_box.v[2] - min_box.v[2]));
  //  float max_bound = std::max(max_box.v[0], std::max(max_box.v[1], max_box.v[2]));
  dx = max_diff / (dim - 2*padding);
  std::cout << "Resolution: " << dx << " with real dimension " << max_diff << std::endl;
  Vec3ui sizes = Vec3ui(dim);

  // remap the min bounds so that the object is centered
  Vec3f center = min_box + (max_box - min_box) / 2;
  std::cout << "Center of grid " << center << std::endl;
  min_box = center - Vec3f((dim * dx) / 2);

  //  min_box -= padding*dx*unit;

  std::cout << "Bound box size: (" << min_box << ") to (" << max_box << ") with dimensions " << sizes << "." << std::endl;

  std::cout << "Computing signed distance field.\n";
  Array3f phi_grid;
  make_level_set3(faceList, vertList, min_box, dx, sizes[0], sizes[1], sizes[2], phi_grid);

  //Very hackily strip off file suffix.
  std::string outname = filename.substr(0, filename.size()-4) + std::string(".sdf");
  std::cout << "Writing results to: " << outname << "\n";
  std::ofstream outfile( outname.c_str());
  outfile << phi_grid.ni << " " << phi_grid.nj << " " << phi_grid.nk << std::endl;
  outfile << min_box[0] << " " << min_box[1] << " " << min_box[2] << std::endl;
  outfile << dx << std::endl;
  for(unsigned int i = 0; i < phi_grid.a.size(); ++i) {
    outfile << phi_grid.a[i] << std::endl;
  }
  outfile.close();
  std::cout << "Processing complete.\n";

return 0;
}
