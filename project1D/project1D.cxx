#include <iostream>
#include <vtkDataSet.h>
#include <vtkImageData.h>
#include <vtkPNGWriter.h>
#include <vtkPointData.h>
#include <vtkPolyData.h>
#include <vtkPolyDataReader.h>
#include <vtkPoints.h>
#include <vtkUnsignedCharArray.h>
#include <vtkFloatArray.h>
#include <vtkCellArray.h>

using std::cerr;
using std::endl;

double ceil_441(double f)
{
    return ceil(f-0.00001);
}

double floor_441(double f)
{
    return floor(f+0.00001);
}


vtkImageData *
NewImage(int width, int height)
{
    vtkImageData *img = vtkImageData::New();
    img->SetDimensions(width, height, 1);
    img->AllocateScalars(VTK_UNSIGNED_CHAR, 3);

    return img;
}

void
WriteImage(vtkImageData *img, const char *filename)
{
   std::string full_filename = filename;
   full_filename += ".png";
   vtkPNGWriter *writer = vtkPNGWriter::New();
   writer->SetInputData(img);
   writer->SetFileName(full_filename.c_str());
   writer->Write();
   writer->Delete();
}

class Triangle
{
  public:
      double         X[3];
      double         Y[3];
      double         Z[3];
	  double         colors[3][3];
  // would some methods for transforming the triangle in place be helpful?
};

class Screen
{
  public:
      unsigned char   *buffer;
      int width, height;
      double *z_buffer;


  // would some methods for accessing and setting pixels be helpful?
};

std::vector<Triangle>
GetTriangles(void)
{
    vtkPolyDataReader *rdr = vtkPolyDataReader::New();
    rdr->SetFileName("proj1d_geometry.vtk");
    cerr << "Reading" << endl;
    rdr->Update();
    cerr << "Done reading" << endl;
    if (rdr->GetOutput()->GetNumberOfCells() == 0)
    {
        cerr << "Unable to open file!!" << endl;
        exit(EXIT_FAILURE);
    }
    vtkPolyData *pd = rdr->GetOutput();
    int numTris = pd->GetNumberOfCells();
    vtkPoints *pts = pd->GetPoints();
    vtkCellArray *cells = pd->GetPolys();
    vtkFloatArray *var = (vtkFloatArray *) pd->GetPointData()->GetArray("hardyglobal");
    float *color_ptr = var->GetPointer(0);
    std::vector<Triangle> tris(numTris);
    vtkIdType npts;
    vtkIdType *ptIds;
    int idx;
    for (idx = 0, cells->InitTraversal() ; cells->GetNextCell(npts, ptIds) ; idx++)
    {
        if (npts != 3)
        {
            cerr << "Non-triangles!! ???" << endl;
            exit(EXIT_FAILURE);
        }
        tris[idx].X[0] = pts->GetPoint(ptIds[0])[0];
        tris[idx].X[1] = pts->GetPoint(ptIds[1])[0];
        tris[idx].X[2] = pts->GetPoint(ptIds[2])[0];
        tris[idx].Y[0] = pts->GetPoint(ptIds[0])[1];
        tris[idx].Y[1] = pts->GetPoint(ptIds[1])[1];
        tris[idx].Y[2] = pts->GetPoint(ptIds[2])[1];
        tris[idx].Z[0] = pts->GetPoint(ptIds[0])[2];
        tris[idx].Z[1] = pts->GetPoint(ptIds[1])[2];
        tris[idx].Z[2] = pts->GetPoint(ptIds[2])[2];
        // 1->2 interpolate between light blue, dark blue
        // 2->2.5 interpolate between dark blue, cyan
        // 2.5->3 interpolate between cyan, green
        // 3->3.5 interpolate between green, yellow
        // 3.5->4 interpolate between yellow, orange
        // 4->5 interpolate between orange, brick
        // 5->6 interpolate between brick, salmon
        double mins[7] = { 1, 2, 2.5, 3, 3.5, 4, 5 };
        double maxs[7] = { 2, 2.5, 3, 3.5, 4, 5, 6 };
        unsigned char RGB[8][3] = { { 71, 71, 219 }, 
                                    { 0, 0, 91 },
                                    { 0, 255, 255 },
                                    { 0, 128, 0 },
                                    { 255, 255, 0 },
                                    { 255, 96, 0 },
                                    { 107, 0, 0 },
                                    { 224, 76, 76 } 
                                  };
        for (int j = 0 ; j < 3 ; j++)
        {
            float val = color_ptr[ptIds[j]];
            int r;
            for (r = 0 ; r < 7 ; r++)
            {
                if (mins[r] <= val && val < maxs[r])
                    break;
            }
            if (r == 7)
            {
                cerr << "Could not interpolate colors for " << val << endl;
                exit(EXIT_FAILURE);
            }
            double proportion = (val-mins[r]) / (maxs[r]-mins[r]);
            tris[idx].colors[j][0] = (RGB[r][0]+proportion*(RGB[r+1][0]-RGB[r][0]))/255.0;
            tris[idx].colors[j][1] = (RGB[r][1]+proportion*(RGB[r+1][1]-RGB[r][1]))/255.0;
            tris[idx].colors[j][2] = (RGB[r][2]+proportion*(RGB[r+1][2]-RGB[r][2]))/255.0;
        }
    }

    return tris;
}


using namespace std; 
Triangle
basic_setup(Triangle tri) {
    // sort the y value and use this as indicator
    // about sort a pair credit: https://www.geeksforgeeks.org/sorting-vector-of-pairs-in-c-set-1-sort-by-first-and-second/
    std::vector<double>label_xy(11); 
    vector< pair <double,int> > tri_originY;
    tri_originY.push_back(make_pair(tri.Y[0],0));
    tri_originY.push_back(make_pair(tri.Y[1],1));
    tri_originY.push_back(make_pair(tri.Y[2],2));
  
    // Using simple sort() function to sort 
    sort(tri_originY.begin(), tri_originY.end());
    /*
    The vector after sort operation is: 
    value   index
    877.551  1
    877.551  2
    897.959  0
    */
    int rowMinI,splitI,rowMaxI;
    if (tri_originY[0].first == tri_originY[1].first) { // first 2 => minY=>goingUp triangle
        rowMinI = tri_originY[0].second;
        splitI = tri_originY[1].second;
        rowMaxI = tri_originY[2].second;
        if (tri.X[rowMinI] > tri.X[splitI]) { //no matter Up or Down triangle, rowMin/rowMax always on the left side (smaller x value)
            // swap two indexes, 
            int temp = rowMinI;
            rowMinI = splitI;
            splitI = temp;
        }
    }

    else if (tri_originY[1].first == tri_originY[2].first) { // last 2 => maxY=>goingDown triangle
        rowMaxI = tri_originY[1].second;
        splitI = tri_originY[2].second;
        rowMinI = tri_originY[0].second;
        if (tri.X[rowMaxI] > tri.X[splitI]) { //no matter Up or Down triangle, rowMin/rowMax always on the left side (smaller x value)
            // swap two indexes, 
            int temp = rowMaxI;
            rowMaxI = splitI;
            splitI = temp;
        }
    }

    else { // arbitrary triangle
        rowMinI = tri_originY[0].second;
        splitI = tri_originY[1].second;
        rowMaxI = tri_originY[2].second;
    }

    int indexs[3] = {rowMinI,splitI,rowMaxI};

    Triangle sorted_triangle;
    for (int i=0;i<3;i++) {
        sorted_triangle.X[i] = tri.X[indexs[i]];
        sorted_triangle.Y[i] = tri.Y[indexs[i]];
        sorted_triangle.Z[i] = tri.Z[indexs[i]];
        sorted_triangle.colors[i][0] = tri.colors[indexs[i]][0];
        sorted_triangle.colors[i][1] = tri.colors[indexs[i]][1];
        sorted_triangle.colors[i][2] = tri.colors[indexs[i]][2];
    }
    return sorted_triangle;

}

double 
LERP (double t,double fa,double fb) {
    //double t = (x-a) / (b-a);
    double fx = fa + t*(fb-fa);
    return fx;
}

vector<Triangle>
split_tri(Triangle triangle) {
    // 0:rowMin, 1:split, 2:rowMax
    // find the intersect vertex, in the line connect rowMin and rowMax, so A:rowMin(0), B:rowMax(2)
    double t=0;
    if (triangle.Y[2] != triangle.Y[0]) t = (triangle.Y[1] - triangle.Y[0]) / (triangle.Y[2] - triangle.Y[0]);
    double intersectX,intersectZ,intersectR,intersectG,intersectB;
    intersectX = LERP(t,triangle.X[0],triangle.X[2]);
    intersectZ = LERP(t,triangle.Z[0],triangle.Z[2]);
    intersectR = LERP(t,triangle.colors[0][0],triangle.colors[2][0]);
    intersectG = LERP(t,triangle.colors[0][1],triangle.colors[2][1]);
    intersectB = LERP(t,triangle.colors[0][2],triangle.colors[2][2]);

    Triangle uptri,downtri;
    // generate up triangle
    for (int i=1;i<3;i++) {
        //intersectX is on the right side, should be the new split point
        uptri.X[i] = triangle.X[i];
        uptri.Y[i] = triangle.Y[i];
        uptri.Z[i] = triangle.Z[i];
        uptri.colors[i][0] = triangle.colors[i][0];
        uptri.colors[i][1] = triangle.colors[i][1];
        uptri.colors[i][2] = triangle.colors[i][2];
    }
    uptri.X[0] = intersectX;
    uptri.Y[0] = triangle.Y[1];
    uptri.Z[0] = intersectZ;
    uptri.colors[0][0] = intersectR;
    uptri.colors[0][1] = intersectG;
    uptri.colors[0][2] = intersectB;

    // generate down triangle
    for (int i=0;i<2;i++) {
        //intersectX is on the right side, should be the new split point
        downtri.X[i] = triangle.X[i];
        downtri.Y[i] = triangle.Y[i];
        downtri.Z[i] = triangle.Z[i];
        downtri.colors[i][0] = triangle.colors[i][0];
        downtri.colors[i][1] = triangle.colors[i][1];
        downtri.colors[i][2] = triangle.colors[i][2];
    }
    downtri.X[2] = intersectX;
    downtri.Y[2] = triangle.Y[1];
    downtri.Z[2] = intersectZ;
    downtri.colors[2][0] = intersectR;
    downtri.colors[2][1] = intersectG;
    downtri.colors[2][2] = intersectB;

    vector<Triangle> UDTriangle (2);
    UDTriangle[0] = uptri;
    UDTriangle[1] = downtri;

    return UDTriangle;
}

void 
fill_up (vtkImageData *image,unsigned char *buffer,int width, int height,double *z_buffer,Triangle tri) {
    //cout<<"Rasterizing GoingUpTriangle"<<endl;
    //cout<<"Triangle: "<<endl;
    //cout << "    Vertex 0: position = ("<<tri.X[0]<<", "<<tri.Y[0]<<", "<<tri.Z[0]<<"), colors = ("<<tri.colors[0][0]<<", "<<tri.colors[0][1]<<", "<<tri.colors[0][2]<<")"<<endl;
    //cout << "    Vertex 1: position = ("<<tri.X[1]<<", "<<tri.Y[1]<<", "<<tri.Z[1]<<"), colors = ("<<tri.colors[1][0]<<", "<<tri.colors[1][1]<<", "<<tri.colors[1][2]<<")"<<endl;
    //cout << "    Vertex 2: position = ("<<tri.X[2]<<", "<<tri.Y[2]<<", "<<tri.Z[2]<<"), colors = ("<<tri.colors[2][0]<<", "<<tri.colors[2][1]<<", "<<tri.colors[2][2]<<")"<<endl;   

    double rowMin = tri.Y[0];
    double rowMax = tri.Y[2];
    if (floor_441(rowMax) - ceil_441(rowMin) >= 0) { // at least in the same row
        double leftendT = 0;
        double leftendX,leftendZ;
        double leftendR,leftendG,leftendB;

        double rightendT = 0;
        double rightendX,rightendZ;
        double rightendR,rightendG,rightendB;
        for (int y=ceil_441(rowMin); y<=floor_441(rowMax);y++) {
            if (rowMax != rowMin) 
                leftendT = (y-rowMin) / (rowMax - rowMin);
            leftendX = LERP(leftendT,tri.X[0],tri.X[2]);
            leftendZ = LERP(leftendT,tri.Z[0],tri.Z[2]);
            leftendR = LERP(leftendT,tri.colors[0][0],tri.colors[2][0]);
            leftendG = LERP(leftendT,tri.colors[0][1],tri.colors[2][1]);
            leftendB = LERP(leftendT,tri.colors[0][2],tri.colors[2][2]);

            if (rowMax != tri.Y[1]) 
                rightendT = (y-tri.Y[1]) / (rowMax - tri.Y[1]);
            rightendX = LERP(rightendT,tri.X[1],tri.X[2]);
            rightendZ = LERP(rightendT,tri.Z[1],tri.Z[2]);
            rightendR = LERP(rightendT,tri.colors[1][0],tri.colors[2][0]);
            rightendG = LERP(rightendT,tri.colors[1][1],tri.colors[2][1]);
            rightendB = LERP(rightendT,tri.colors[1][2],tri.colors[2][2]);

            if ( floor_441(rightendX) - ceil_441(leftendX) >= 0) {
                // Rastering along row 873 with left end = 551.02 (Z: -0.948154, RGB = 0.11181/0.11181/0.558437) and right end = 557.029 (Z: -0.949022, RGB = 0.112817/0.112817/0.560251)
                //cout<<"Rastering along row "<<y<<" with left end = "<<leftendX<<" (Z: "<<leftendZ<<", RGB = "<<leftendR<<"/"<<leftendG<<"/"<<leftendB<<") and right end = "<<rightendX<<" (Z: "<<rightendZ<<", RGB = "<<rightendR<<"/"<<rightendG<<"/"<<rightendB<<")" <<endl;

                for (int x=ceil_441(leftendX); x<=floor_441(rightendX); x++) {
                    double t = 0;
                    double z;
                    if (rightendX != leftendX) 
                        t = (x-leftendX) / (rightendX - leftendX);
                    z = LERP(t,leftendZ,rightendZ);
                    //cout<<"------ t = "<<t<<", z = "<<z<<endl;
                    if (x<0 || x>=width || y<0 || y>=height) {
                        continue;
                    }
                    if (z < z_buffer[y*width+x]) {
                        continue;
                    }
                    
                    else {
                        z_buffer[y*width+x] = z;
                        double R = LERP(t,leftendR,rightendR);
                        double G = LERP(t,leftendG,rightendG);
                        double B = LERP(t,leftendB,rightendB);
                        //cout<<"    Got fragment r = "<<y<<", c = "<<x<<", z = "<<z<<", color = "<<R<<"/"<<G<<"/"<<B<<endl;
                        //cout<<"\n"<<endl;

                        buffer = (unsigned char *) image->GetScalarPointer(x,y,0);
                        buffer[0] = ceil_441(R*255);
                        buffer[1] = ceil_441(G*255);
                        buffer[2] = ceil_441(B*255);
                        /*
                        buffer[y*width+x] = ceil_441(R*255);
                        buffer[y*width+x+1] = ceil_441(G*255);
                        buffer[y*width+x+2] = ceil_441(B*255);
                        */
                    }  
                }
            } 
        }
    }
}


void 
fill_down (vtkImageData *image,unsigned char *buffer,int width, int height,double *z_buffer,Triangle tri) {
    //cout<<"Rasterizing GoingDownTriangle"<<endl;
    //cout<<"Triangle: "<<endl;
    //cout << "    Vertex 0: position = ("<<tri.X[0]<<", "<<tri.Y[0]<<", "<<tri.Z[0]<<"), colors = ("<<tri.colors[0][0]<<", "<<tri.colors[0][1]<<", "<<tri.colors[0][2]<<")"<<endl;
    //cout << "    Vertex 1: position = ("<<tri.X[1]<<", "<<tri.Y[1]<<", "<<tri.Z[1]<<"), colors = ("<<tri.colors[1][0]<<", "<<tri.colors[1][1]<<", "<<tri.colors[1][2]<<")"<<endl;
    //cout << "    Vertex 2: position = ("<<tri.X[2]<<", "<<tri.Y[2]<<", "<<tri.Z[2]<<"), colors = ("<<tri.colors[2][0]<<", "<<tri.colors[2][1]<<", "<<tri.colors[2][2]<<")"<<endl;   

    double rowMin = tri.Y[0];
    double rowMax = tri.Y[2];
    if (floor_441(rowMax) - ceil_441(rowMin) >= 0) { // at least in the same row
        double leftendT = 0;
        double leftendX,leftendZ;
        double leftendR,leftendG,leftendB;

        double rightendT = 0;
        double rightendX,rightendZ;
        double rightendR,rightendG,rightendB;
        for (int y=ceil_441(rowMin); y<=floor_441(rowMax);y++) {
            if (rowMax != rowMin) 
                leftendT = (y-rowMin) / (rowMax - rowMin);
            leftendX = LERP(leftendT,tri.X[0],tri.X[2]);
            leftendZ = LERP(leftendT,tri.Z[0],tri.Z[2]);
            leftendR = LERP(leftendT,tri.colors[0][0],tri.colors[2][0]);
            leftendG = LERP(leftendT,tri.colors[0][1],tri.colors[2][1]);
            leftendB = LERP(leftendT,tri.colors[0][2],tri.colors[2][2]);

            if (tri.Y[1] != tri.Y[0]) 
                rightendT = (y-tri.Y[0]) / (tri.Y[1] - tri.Y[0]);
            rightendX = LERP(rightendT,tri.X[0],tri.X[1]);
            rightendZ = LERP(rightendT,tri.Z[0],tri.Z[1]);
            rightendR = LERP(rightendT,tri.colors[0][0],tri.colors[1][0]);
            rightendG = LERP(rightendT,tri.colors[0][1],tri.colors[1][1]);
            rightendB = LERP(rightendT,tri.colors[0][2],tri.colors[1][2]);

            if ( floor_441(rightendX) - ceil_441(leftendX) >= 0) {
                //cout<<"Rastering along row "<<y<<" with left end = "<<leftendX<<" (Z: "<<leftendZ<<", RGB = "<<leftendR<<"/"<<leftendG<<"/"<<leftendB<<") and right end = "<<rightendX<<" (Z: "<<rightendZ<<", RGB = "<<rightendR<<"/"<<rightendG<<"/"<<rightendB<<")" <<endl;

                for (int x=ceil_441(leftendX); x<=floor_441(rightendX); x++) {
                    double t=0;
                    double z;
                    if (rightendX != leftendX) 
                        t = (x-leftendX) / (rightendX - leftendX);
                    z = LERP(t,leftendZ,rightendZ);
                    //cout<<"------ t = "<<t<<", z = "<<z<<endl;
                    if (x<0 || x>=width || y<0 || y>=height) {
                        continue;
                    }
                    if (z < z_buffer[y*width+x]) {
                        continue;
                    }
                    
                    else {
                        z_buffer[y*width+x] = z;
                        double R = LERP(t,leftendR,rightendR);
                        double G = LERP(t,leftendG,rightendG);
                        double B = LERP(t,leftendB,rightendB);
                        //cout<<"    Got fragment r = "<<y<<", c = "<<x<<", z = "<<z<<", color = "<<R<<"/"<<G<<"/"<<B<<endl;
                        //cout<<"\n"<<endl;
                        buffer = (unsigned char *) image->GetScalarPointer(x,y,0);
                        buffer[0] = ceil_441(R*255);
                        buffer[1] = ceil_441(G*255);
                        buffer[2] = ceil_441(B*255);
                        /*
                        buffer[y*width+x] = ceil_441(R*255);
                        buffer[y*width+x+1] = ceil_441(G*255);
                        buffer[y*width+x+2] = ceil_441(B*255);
                        */
                    }  
                }
            }
        }
    }
}


int main()
{
    vtkImageData *image = NewImage(1000, 1000);
    unsigned char *buffer = 
     (unsigned char *) image->GetScalarPointer(0,0,0);
    int npixels = 1000*1000;
    for (int i = 0 ; i < npixels*3 ; i++)
       buffer[i] = 0;
   
   Screen screen;
   screen.buffer = buffer;
   screen.width = 1000;
   screen.height = 1000;
   screen.z_buffer = new double [1000*1000];
   for (int i=0;i<1000*1000;i++){
       screen.z_buffer[i] = -1;
   }
   

   vector<Triangle> triangles = GetTriangles();
   int tri_size = triangles.size();

   // YOUR CODE GOES HERE TO DEPOSIT THE COLORS FROM TRIANGLES 
   // INTO PIXELS USING THE SCANLINE ALGORITHM
   // tri_size
    for (int t=0;t<tri_size;t++) { 
        // 1, basic_setup, label left, middle, right x, y and z
        //cout<<"Working on triangle "<<t<<endl;
        Triangle sort_tri = basic_setup(triangles[t]);

        if (sort_tri.Y[0] == sort_tri.Y[1]) {
            fill_up (image,buffer,screen.width,screen.height,screen.z_buffer,sort_tri);
        }
        else if (sort_tri.Y[1] == sort_tri.Y[2]) {
            fill_down (image,buffer,screen.width,screen.height,screen.z_buffer,sort_tri);
        }
        else {
            vector<Triangle> UDTriangle = split_tri(sort_tri);
            Triangle Up_tri = basic_setup(UDTriangle[0]);
            fill_up (image,buffer,screen.width,screen.height,screen.z_buffer,Up_tri);
            Triangle Down_tri = basic_setup(UDTriangle[1]);
            fill_down (image,buffer,screen.width,screen.height,screen.z_buffer,Down_tri);
        }
        //cout <<"End routine for "<<t<<endl;
        //cout <<"---------------"<<endl;   
    }

   WriteImage(image, "allTriangles");
}
