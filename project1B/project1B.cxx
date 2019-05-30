#include <iostream>
#include <vtkDataSet.h>
#include <vtkImageData.h>
#include <vtkPNGWriter.h>
#include <vtkPointData.h>

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
   std::vector<Triangle> rv(100);

   unsigned char colors[6][3] = { {255,128,0}, {255, 0, 127}, {0,204,204}, 
                                  {76,153,0}, {255, 204, 204}, {204, 204, 0}};
   for (int i = 0 ; i < 100 ; i++)
   {
       int idxI = i%10;
       int posI = idxI*100;
       int idxJ = i/10;
       int posJ = idxJ*100;
       int firstPt = (i%3);
       rv[i].X[firstPt] = posI;
       if (i == 50)
           rv[i].X[firstPt] = -10;
       rv[i].Y[firstPt] = posJ+10*(idxJ+1);
       rv[i].X[(firstPt+1)%3] = posI+99;
       rv[i].Y[(firstPt+1)%3] = posJ+10*(idxJ+1);
       rv[i].X[(firstPt+2)%3] = posI+i;
       rv[i].Y[(firstPt+2)%3] = posJ;
       if (i == 5)
          rv[i].Y[(firstPt+2)%3] = -50;
       rv[i].color[0] = colors[i%6][0];
       rv[i].color[1] = colors[i%6][1];
       rv[i].color[2] = colors[i%6][2];
   }

   return rv;
}


void
fill_color (vtkImageData *image, unsigned char *buffer,int width, int height) {
    std::vector<Triangle> triangles = GetTriangles();
    for (int t=0; t<100; t++) {
        double lx,mx,rx,ly,my,ry;

        // determing the x of each vertex
        //case 1, first two in same line
        if (triangles[t].Y[0] == triangles[t].Y[1]) {
            cout << "case1: y0 = y1"<<endl;
            lx = std::min(triangles[t].X[0],triangles[t].X[1]);
            rx = std::max(triangles[t].X[0],triangles[t].X[1]);
            ly = ry = triangles[t].Y[0];

            mx = triangles[t].X[2];
            my = triangles[t].Y[2];
        }
        //case 2, first and thrid in same line
        if (triangles[t].Y[0] == triangles[t].Y[2]) {
            cout << "case2: y0 = y2"<<endl;
            lx = std::min(triangles[t].X[0],triangles[t].X[2]);
            rx = std::max(triangles[t].X[0],triangles[t].X[2]);
            ly = ry = triangles[t].Y[0];

            mx = triangles[t].X[1];
            my = triangles[t].Y[1];
        }
        //case 3, first is middle
        if (triangles[t].Y[1] == triangles[t].Y[2]) {
            cout << "case3: y1 = y2"<<endl;
            mx = triangles[t].X[0];
            my = triangles[t].Y[0];

            lx = std::min(triangles[t].X[1],triangles[t].X[2]);
            rx = std::max(triangles[t].X[1],triangles[t].X[2]);
            ly = ry = triangles[t].Y[1];
        }
    
        /*
            k = (y2-y1)/(x2-x1)
            b = y - kx
        */
        //calculate left and middle vertex line
        double lk,rk,lb,rb;
        if ( (mx-lx) == 0 ){
            lk=0;
        }
        else {
            lk = (my-ly) / (mx-lx);
        }
        lb = ly - lk*lx;

        if ( (rx-mx)==0 ){
            rk = 0;
        }
        else {
            rk = (ry-my) / (rx-mx);
        }
        rb = ry - rk*rx;

        // y = kx + b => x = (y-b)/k
        for (int y=ceil_441(my); y<=floor_441(ly);y++) {
            double leftEnd, rightEnd;
            if (lk==0) {
                leftEnd = lx;
            }
            else {
                leftEnd = (y-lb) / lk;  
            }
            
            if (rk==0) {
                rightEnd = rx;
            }
            else {
                rightEnd = (y-rb) / rk;
            }
            
            for (int x=ceil_441(leftEnd); x<=floor_441(rightEnd); x++) {
                if (x<0 || x>=width || y<0 || y>=height)
                    continue;
                buffer = (unsigned char *) image->GetScalarPointer(x,y,0);
                buffer[0] = triangles[t].color[0];
                buffer[1] = triangles[t].color[1];
                buffer[2] = triangles[t].color[2];
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

   // YOUR CODE GOES HERE TO DEPOSIT THE COLORS FROM TRIANGLES 
   // INTO PIXELS USING THE SCANLINE ALGORITHM

    fill_color (image,buffer,screen.width,screen.height);

   WriteImage(image, "allTriangles");
}
