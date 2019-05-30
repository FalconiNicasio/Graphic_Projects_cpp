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
      unsigned char color[3];

  // would some methods for transforming the triangle in place be helpful?
};

class Screen
{
  public:
      unsigned char   *buffer;
      int width, height;

  // would some methods for accessing and setting pixels be helpful?
};

std::vector<Triangle>
GetTriangles(void)
{
    vtkPolyDataReader *rdr = vtkPolyDataReader::New();
    rdr->SetFileName("proj1c_geometry.vtk");
    cerr << "Reading" << endl;
    rdr->Update();
    if (rdr->GetOutput()->GetNumberOfCells() == 0)
    {
        cerr << "Unable to open file!!" << endl;
        exit(EXIT_FAILURE);
    }
    vtkPolyData *pd = rdr->GetOutput();
    int numTris = pd->GetNumberOfCells();
    vtkPoints *pts = pd->GetPoints();
    vtkCellArray *cells = pd->GetPolys();
    vtkFloatArray *colors = (vtkFloatArray *) pd->GetPointData()->GetArray("color_nodal");
    float *color_ptr = colors->GetPointer(0);
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
        tris[idx].color[0] = (unsigned char) color_ptr[4*ptIds[0]+0];
        tris[idx].color[1] = (unsigned char) color_ptr[4*ptIds[0]+1];
        tris[idx].color[2] = (unsigned char) color_ptr[4*ptIds[0]+2];
    }
    cerr << "Done reading" << endl;

    return tris;
}

std::vector<double> 
basic_setup(Triangle tri) {
    
    double x_origin[] = {tri.X[0],tri.X[1],tri.X[2]};
    double x_arr[] = {tri.X[0],tri.X[1],tri.X[2]};
    double y_origin[] = {tri.Y[0],tri.Y[1],tri.Y[2]};
    int n = sizeof(x_arr)/sizeof(x_arr[0]);
    std::sort(x_arr, x_arr+n);
    double y_arr[3];
    for (int i=0;i<3;i++) {
        int y_index = std::distance(x_origin, std::find(x_origin, x_origin + 3, x_arr[i]));
        y_arr[i] = y_origin[y_index];
    }
    
    // lx,mx,rx,ly,my,ry
    std::vector<double>label_xy(8);
    label_xy[0] =x_arr[0];
    label_xy[1] =x_arr[1];
    label_xy[2] =x_arr[2];
    label_xy[3] =y_arr[0];
    label_xy[4] =y_arr[1];
    label_xy[5] =y_arr[2];
    double rowMin = y_arr[0];
    double rowMax = y_arr[1];
    for (int i=3;i<6;i++){
        if (label_xy[i] < rowMin) 
            rowMin = label_xy[i];
        else if (label_xy[i] > rowMax)
            rowMax = label_xy[i];
    }
    label_xy[6] =rowMin;
    label_xy[7] =rowMax;
    return label_xy;
}


void 
fill_down (Triangle tri,vtkImageData *image, unsigned char *buffer,int width, int height, std::vector<double> label_xy) {
    //cout <<"\n"<<endl;
    //cout << "Rasterizing GoingDownTriangle"<<endl;
    double rowMin = label_xy[6];
    double rowMax = label_xy[7];
    if ( floor_441(rowMax) - ceil_441(rowMin) < 0) {
        
        //cout << "No scanlines: lowY = "<<ceil_441(rowMin)<<", hiY = "<<floor_441(rowMax)<<endl;
        //continue;
    } // 3 vertax are in the same pixel, continue
    // 0-2: x; 3-5:y; 6: rowMin, 7:rowMax;
    else {
        //cout << "Scanlines go from lowY = "<<ceil_441(rowMin)<<", hiY = "<<floor_441(rowMax)<<endl;
        double flat_lx,flat_rx,top_x;
        std::vector<double> end = {};
        //x.push_back("d");
        for (int i=3;i<6;i++) {
            if (label_xy[i] == rowMin) {  //change this for fill_up
                top_x = label_xy[i%3]; //get cooresponding x
            }
            else if (label_xy[i] == rowMax) { //change this for fill_up
                end.push_back(label_xy[i%3]);
            }
        }
        flat_lx = std::min(end[0],end[1]);
        flat_rx = std::max(end[0],end[1]);
        //cout <<"rowMin: "<<rowMin<<", rowMax: "<<rowMax<<", flat_lx: "<<flat_lx<<", flat_rx: "<<flat_rx<<", top_x: "<<top_x<<endl;

        double lk,rk,lb,rb;
        if ( (top_x-flat_lx) == 0 ){
            lk=0;
        }
        else {
            lk = (rowMax - rowMin) / (flat_lx-top_x);
        }
        lb = rowMax - lk*flat_lx;

        if ( (top_x-flat_lx)==0 ){
            rk = 0;
        }
        else {
            rk = (rowMax - rowMin) / (flat_rx-top_x);
        }
        rb = rowMax - rk*flat_rx;

        // y = kx + b => x = (y-b)/k
        for (int y=ceil_441(rowMin); y<=floor_441(rowMax);y++) {
            //cout << "Operating on scanline " << y<<endl;
            double leftEnd, rightEnd;
            if (lk==0) {
                leftEnd = flat_lx;
            }
            else {
                leftEnd = (y-lb) / lk;  
            }
            
            if (rk==0) {
                rightEnd = flat_rx;
            }
            else {
                rightEnd = (y-rb) / rk;
            }

            if ( floor_441(rightEnd) - ceil_441(leftEnd)<0) {  //  && (floor_441(rx)-ceil_441(lx)<=0)
                //cout << "No Fragments: lowX = "<< ceil_441(leftEnd)<<", hiX = "<<floor_441(rightEnd)<<endl;
                //continue;
            } // 3 vertax are in the same pixel, continue
            else {
                //cout << "Fragments go from lowX = "<<ceil_441(leftEnd)<<", hiY = "<<floor_441(rightEnd)<<endl;
                // todo: check rest
                for (int x=ceil_441(leftEnd); x<=floor_441(rightEnd); x++) {
                    
                    if (x<0 || x>=width || y<0 || y>=height) {
                        continue;
                    }
                    //cout <<"Triangle is writing to row "<<y<<", column "<<x<<endl;
                    buffer = (unsigned char *) image->GetScalarPointer(x,y,0);
                    buffer[0] = tri.color[0];
                    buffer[1] = tri.color[1];
                    buffer[2] = tri.color[2];
                }
            }
        }
    }
    
}


void 
fill_up (Triangle tri,vtkImageData *image, unsigned char *buffer,int width, int height, std::vector<double> label_xy) {
    //cout <<"\n"<<endl;
    //cout << "Rasterizing GoingUpTriangle"<<endl;
    double rowMin = label_xy[6];
    double rowMax = label_xy[7];
    if ( floor_441(rowMax) - ceil_441(rowMin) < 0) {
        //cout << "No scanlines: lowY = "<<ceil_441(rowMin)<<", hiY = "<<floor_441(rowMax)<<endl;
        //continue;
    } // 3 vertax are in the same pixel, continue
    // 0-2: x; 3-5:y; 6: rowMin, 7:rowMax;
    else {
        //cout << "Scanlines go from lowY = "<<ceil_441(rowMin)<<", hiY = "<<floor_441(rowMax)<<endl;
        double flat_lx,flat_rx,top_x;
        std::vector<double> end = {};
        //x.push_back("d");
        for (int i=3;i<6;i++) {
            if (label_xy[i] == rowMax) {  //change this for fill_up
                top_x = label_xy[i%3]; //get cooresponding x
            }
            else if (label_xy[i] == rowMin) { //change this for fill_up
                end.push_back(label_xy[i%3]);
            }
        }
        flat_lx = std::min(end[0],end[1]);
        flat_rx = std::max(end[0],end[1]);
        //cout <<"rowMin: "<<rowMin<<", rowMax: "<<rowMax<<", flat_lx: "<<flat_lx<<", flat_rx: "<<flat_rx<<", top_x: "<<top_x<<endl;

        double lk,rk,lb,rb;
        if ( (top_x-flat_lx) == 0 ){
            lk=0;
        }
        else {
            lk = (rowMax - rowMin) / (top_x-flat_lx);  //change for fill_up
        }
        lb = rowMin-lk*flat_lx;

        if ( (top_x-flat_rx)==0 ){
            rk = 0;
        }
        else {
            rk = (rowMax - rowMin) / (top_x-flat_rx);  //change for fill_up
        }
        rb = rowMin-rk*flat_rx;

        // y = kx + b => x = (y-b)/k
        for (int y=ceil_441(rowMin); y<=floor_441(rowMax);y++) { //change for fill_up
            //cout << "Operating on scanline " << y<<endl;
            double leftEnd, rightEnd;
            if (lk==0) {
                leftEnd = flat_lx;
            }
            else {
                leftEnd = (y-lb) / lk;  
            }
            
            if (rk==0) {
                rightEnd = flat_rx;
            }
            else {
                rightEnd = (y-rb) / rk;
            }

            if ( floor_441(rightEnd) - ceil_441(leftEnd) <0) {  //  && (floor_441(rx)-ceil_441(lx)<=0)
                //cout << "No Fragments: lowX = "<< ceil_441(leftEnd)<<", hiY = "<<floor_441(rightEnd)<<endl;
                //continue;
            } // 3 vertax are in the same pixel, continue
            else {
                //cout << "Fragments go from lowX = "<<ceil_441(leftEnd)<<", hiY = "<<floor_441(rightEnd)<<endl;
                // todo: check rest
                for (int x=ceil_441(leftEnd); x<=floor_441(rightEnd); x++) {
                    
                    if (x<0 || x>=width || y<0 || y>=height) {
                        continue;
                    }
                    //cout <<"Triangle is writing to row "<<y<<", column "<<x<<endl;
                    buffer = (unsigned char *) image->GetScalarPointer(x,y,0);
                    buffer[0] = tri.color[0];
                    buffer[1] = tri.color[1];
                    buffer[2] = tri.color[2];
                }
            }
        }
    }
    
}


std::vector<Triangle>
split_tri(Triangle triangle) {
    std::vector<double> label_xy = basic_setup(triangle); // lx,mx,rx,ly,my,ry,rowMin,rowMax
    int rowMin_index,rowMax_index;
    int split_index = 3; //initialize split_index as the right, if this is not been modified, means this is flat-end triangle
    // wait, this does not make sense since all the triangle pass to this function should be arbitrary.
    for (int i=3;i<6;i++) {
        if (label_xy[i] == label_xy[6]) {
            rowMin_index = i;
        }
        else if (label_xy[i] == label_xy[7]) {
            rowMax_index = i;
        }
        else
            split_index = i;
    }
    double k,b;
    if ( label_xy[rowMax_index%3] - label_xy[rowMin_index%3] == 0 ){
        k=0;
    }
    else {
        k = (label_xy[rowMax_index] - label_xy[rowMin_index]) / (label_xy[rowMax_index%3] - label_xy[rowMin_index%3]);
    }
    b = label_xy[rowMax_index] - k*label_xy[rowMax_index%3];
    double splitX;
    // y = kx + b => x = (y-b)/k
    splitX = (label_xy[split_index]-b)/k;
    Triangle uptri,downtri;
    uptri.X[0] = label_xy[rowMax_index%3];
    uptri.Y[0] = label_xy[rowMax_index];
    uptri.X[1] = label_xy[split_index%3];
    uptri.Y[1] = label_xy[split_index];
    uptri.X[2] = splitX;
    uptri.Y[2] = label_xy[split_index];
    uptri.color[0] = triangle.color[0];
    uptri.color[1] = triangle.color[1];
    uptri.color[2] = triangle.color[2];

    downtri.X[0] = label_xy[rowMin_index%3];
    downtri.Y[0] = label_xy[rowMin_index];
    downtri.X[1] = label_xy[split_index%3];
    downtri.Y[1] = label_xy[split_index];
    downtri.X[2] = splitX;
    downtri.Y[2] = label_xy[split_index];
    downtri.color[0] = triangle.color[0];
    downtri.color[1] = triangle.color[1];
    downtri.color[2] = triangle.color[2];

    std::vector<Triangle> UDTriangle (2);
    UDTriangle[0] = uptri;
    UDTriangle[1] = downtri;

    
    //cout << "Triangle: ("<<std::setprecision(13)<<triangle.X[0]<<", "<<triangle.Y[0]<<"), ("<<triangle.X[1]<<", "<<triangle.Y[1]<<"), ("<<triangle.X[2]<<", "<<triangle.Y[2]<<") "<<endl;
    //cout <<"Going up triangle: "<<endl;
    //cout << "         Triangle: ("<<uptri.X[0]<<", "<<uptri.Y[0]<<"), ("<<uptri.X[1]<<", "<<uptri.Y[1]<<"), ("<<uptri.X[2]<<", "<<uptri.Y[2]<<") "<<endl;
    //cout <<"Going down downtri: "<<endl;
    //cout << "         Triangle: ("<<downtri.X[0]<<", "<<downtri.Y[0]<<"), ("<<downtri.X[1]<<", "<<downtri.Y[1]<<"), ("<<downtri.X[2]<<", "<<downtri.Y[2]<<") "<<endl;
    
    return UDTriangle;
}


int main()
{
    vtkImageData *image = NewImage(1786, 1344);
    unsigned char *buffer = 
     (unsigned char *) image->GetScalarPointer(0,0,0);
    int npixels = 1786*1344;
    for (int i = 0 ; i < npixels*3 ; i++)
       buffer[i] = 0;
   
   Screen screen;
   screen.buffer = buffer;
   screen.width = 1786;
   screen.height = 1344;

   std::vector<Triangle> triangles = GetTriangles();
   int tri_size = triangles.size();
   //cout <<"There are "<<tri_size<< " triangles in vector"<<endl;

   // YOUR CODE GOES HERE TO DEPOSIT THE COLORS FROM TRIANGLES 
   // INTO PIXELS USING THE SCANLINE ALGORITHM
   // tri_size
    for (int t=0;t<tri_size;t++) { 
        // 1, basic_setup, label left, middle, right x and y
        //cout <<"Triangle "<<t<<endl;
        std::vector<double> label_xy = basic_setup(triangles[t]); // lx,mx,rx,ly,my,ry
        
        // 2, determine the angle (flat end/arbitrary)
        //double lx,mx,rx,rowMin,rowMax;
        double ly,my,ry,rowMin,rowMax;
        //lx = label_xy[0];
        //mx = label_xy[1];
        //rx = label_xy[2];
        ly = label_xy[3];
        my = label_xy[4];
        ry = label_xy[5];
        rowMin = label_xy[6];
        rowMax = label_xy[7];
        
        if (ly==my) {
            
            if (ly==rowMin) {
                fill_up (triangles[t],image,buffer,screen.width,screen.height,label_xy);
            }
            
            
            if (ly==rowMax) {
                fill_down (triangles[t],image,buffer,screen.width,screen.height,label_xy);
            }
            
        }
        if (ly==ry) {
            
            if (ly==rowMin) {
                fill_up (triangles[t],image,buffer,screen.width,screen.height,label_xy);
            }
            
            
            if (ly==rowMax) {
                fill_down (triangles[t],image,buffer,screen.width,screen.height,label_xy);
            }
            
        }
        if (my==ry) {
            
            if (my==rowMin) {
                fill_up (triangles[t],image,buffer,screen.width,screen.height,label_xy);
            }
            
            
            if (my==rowMax) {
                fill_down (triangles[t],image,buffer,screen.width,screen.height,label_xy);
            }
            
        }

        else {
            std::vector<Triangle> UDTriangle = split_tri(triangles[t]);
            std::vector<double> label_up = basic_setup(UDTriangle[0]);
            fill_up (UDTriangle[0],image,buffer,screen.width,screen.height,label_up);
            std::vector<double> label_down = basic_setup(UDTriangle[1]);
            fill_down (UDTriangle[1],image,buffer,screen.width,screen.height,label_down);
        }
        
        //cout <<"End routine for "<<t<<endl;
        //cout <<"---------------"<<endl;   
    }

   WriteImage(image, "allTriangles");
}
