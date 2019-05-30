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
#include <vtkDoubleArray.h>
#define	 NORMALS


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

// matrix.cxx
class Matrix
{
  public:
    double          A[4][4];

    void            TransformPoint(const double *ptIn, double *ptOut);
    static Matrix   ComposeMatrices(const Matrix &, const Matrix &);
    void            Print(ostream &o);
};

void
Matrix::Print(ostream &o)
{
    for (int i = 0 ; i < 4 ; i++)
    {
        char str[256];
        sprintf(str, "(%.7f %.7f %.7f %.7f)\n", A[i][0], A[i][1], A[i][2], A[i][3]);
        o << str;
    }
}

Matrix
Matrix::ComposeMatrices(const Matrix &M1, const Matrix &M2)
{
    Matrix rv;
    for (int i = 0 ; i < 4 ; i++)
        for (int j = 0 ; j < 4 ; j++)
        {
            rv.A[i][j] = 0;
            for (int k = 0 ; k < 4 ; k++)
                rv.A[i][j] += M1.A[i][k]*M2.A[k][j];
        }

    return rv;
}

// camera.cxx
class Camera
{
  public:
    double          near, far;
    double          angle;
    double          position[3];
    double          focus[3];
    double          up[3];

    Matrix          ViewTransform(void);
    Matrix          CameraTransform(void);
    Matrix          DeviceTransform(int width,int height);
};


double SineParameterize(int curFrame, int nFrames, int ramp)
{
    int nNonRamp = nFrames-2*ramp;
    double height = 1./(nNonRamp + 4*ramp/M_PI);
    if (curFrame < ramp)
    {
        double factor = 2*height*ramp/M_PI;
        double eval = cos(M_PI/2*((double)curFrame)/ramp);
        return (1.-eval)*factor;
    }
    else if (curFrame > nFrames-ramp)
    {
        int amount_left = nFrames-curFrame;
        double factor = 2*height*ramp/M_PI;
        double eval =cos(M_PI/2*((double)amount_left/ramp));
        return 1. - (1-eval)*factor;
    }
    double amount_in_quad = ((double)curFrame-ramp);
    double quad_part = amount_in_quad*height;
    double curve_part = height*(2*ramp)/M_PI;
    return quad_part+curve_part;
}

Camera
GetCamera(int frame, int nframes)
{
    double t = SineParameterize(frame, nframes, nframes/10);
    Camera c;
    c.near = 5;
    c.far = 200;
    c.angle = M_PI/6;
    c.position[0] = 40*sin(2*M_PI*t);
    c.position[1] = 40*cos(2*M_PI*t);
    c.position[2] = 40;
    c.focus[0] = 0;
    c.focus[1] = 0;
    c.focus[2] = 0;
    c.up[0] = 0;
    c.up[1] = 1;
    c.up[2] = 0;
    return c;
}


// double output_vertex [4];
// Matrix.TransformPoint (vertex,output_vertex)

void
Matrix::TransformPoint(const double *ptIn, double *ptOut)
{
    ptOut[0] = ptIn[0]*A[0][0]
             + ptIn[1]*A[1][0]
             + ptIn[2]*A[2][0]
             + ptIn[3]*A[3][0];
    ptOut[1] = ptIn[0]*A[0][1]
             + ptIn[1]*A[1][1]
             + ptIn[2]*A[2][1]
             + ptIn[3]*A[3][1];
    ptOut[2] = ptIn[0]*A[0][2]
             + ptIn[1]*A[1][2]
             + ptIn[2]*A[2][2]
             + ptIn[3]*A[3][2];
    ptOut[3] = ptIn[0]*A[0][3]
             + ptIn[1]*A[1][3]
             + ptIn[2]*A[2][3]
             + ptIn[3]*A[3][3];
}

class Triangle
{
  public:
      double         X[3];
      double         Y[3];
      double         Z[3];
      double         shading[3];
	  double         colors[3][3];
      double         normals[3][3];
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
    rdr->SetFileName("proj1e_geometry.vtk");
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
    vtkDoubleArray *var = (vtkDoubleArray *) pd->GetPointData()->GetArray("hardyglobal");
    double *color_ptr = var->GetPointer(0);
    //vtkFloatArray *var = (vtkFloatArray *) pd->GetPointData()->GetArray("hardyglobal");
    //float *color_ptr = var->GetPointer(0);
    vtkFloatArray *n = (vtkFloatArray *) pd->GetPointData()->GetNormals();
    float *normals = n->GetPointer(0);
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
        double *pt = NULL;
        pt = pts->GetPoint(ptIds[0]);
        tris[idx].X[0] = pt[0];
        tris[idx].Y[0] = pt[1];
        tris[idx].Z[0] = pt[2];
#ifdef NORMALS
        tris[idx].normals[0][0] = normals[3*ptIds[0]+0];
        tris[idx].normals[0][1] = normals[3*ptIds[0]+1];
        tris[idx].normals[0][2] = normals[3*ptIds[0]+2];
#endif
        pt = pts->GetPoint(ptIds[1]);
        tris[idx].X[1] = pt[0];
        tris[idx].Y[1] = pt[1];
        tris[idx].Z[1] = pt[2];
#ifdef NORMALS
        tris[idx].normals[1][0] = normals[3*ptIds[1]+0];
        tris[idx].normals[1][1] = normals[3*ptIds[1]+1];
        tris[idx].normals[1][2] = normals[3*ptIds[1]+2];
#endif
        pt = pts->GetPoint(ptIds[2]);
        tris[idx].X[2] = pt[0];
        tris[idx].Y[2] = pt[1];
        tris[idx].Z[2] = pt[2];
#ifdef NORMALS
        tris[idx].normals[2][0] = normals[3*ptIds[2]+0];
        tris[idx].normals[2][1] = normals[3*ptIds[2]+1];
        tris[idx].normals[2][2] = normals[3*ptIds[2]+2];
#endif

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
                cerr << "Could not interpolate color for " << val << endl;
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


struct LightingParameters
{
    LightingParameters(void)
    {
         lightDir[0] = -0.6;
         lightDir[1] = 0;
         lightDir[2] = -0.8;
         Ka = 0.3;
         Kd = 0.7;
         Ks = 2.3;
         alpha = 2.5;
    };
  

    double lightDir[3]; // The direction of the light source
    double Ka;           // The coefficient for ambient lighting.
    double Kd;           // The coefficient for diffuse lighting.
    double Ks;           // The coefficient for specular lighting.
    double alpha;        // The exponent term for specular lighting.
};
LightingParameters lp;
// ------------------------- end of setup class and functions

// project 1E New content: 
Screen
InitializeScreen(vtkImageData *image,int w,int h) {
    unsigned char *buffer = 
     (unsigned char *) image->GetScalarPointer(0,0,0);
    int npixels = w*h;
    for (int i = 0 ; i < npixels*3 ; i++)
       buffer[i] = 0;
    
    Screen screen;
    screen.width = w;
    screen.height = h;
    screen.buffer = buffer;

   screen.z_buffer = new double [npixels];
   for (int i=0;i<npixels;i++){
       screen.z_buffer[i] = -1;
   }
   return screen;
}

//Camera
double *
VectorCrossProduct(const double *v1,const double *v2) {
    double * cross_result = new double[3];
    // x:0, y:1, z:2
    cross_result[0] = v1[1]*v2[2] - v1[2]*v2[1];
    cross_result[1] = v1[2]*v2[0] - v1[0]*v2[2];
    cross_result[2] = v1[0]*v2[1] - v1[1]*v2[0];
    return cross_result;
}

void
Normalize (double*v) {
    double normV = sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
    for (int i=0;i<3;i++) {
        v[i] = v[i] / normV;
    }
}

double
VectorDotProduct(const double *v1,double *v2) {
    double dot_result=0;
    for (int i=0;i<3;i++) {
        dot_result += v1[i]*v2[i];
    }
    return dot_result;
}

Matrix 
Camera::CameraTransform() {
    //first three row of matrix: ppt6, slide 46
    double *O = new double [3];
    double *u = new double [3];
    double *v = new double [3];
    double *w = new double [3];
    double *O_focus = new double [3];
    for (int i=0;i<3;i++) {
        O[i] = position[i];
        O_focus[i] = O[i] - focus[i];
    }
    u = VectorCrossProduct(up,O_focus);
    v = VectorCrossProduct(O_focus,u);
    w = O_focus;

    Normalize(u);
    Normalize(v);
    Normalize(w);

    double*t = new double [3];
    for (int j=0;j<3;j++) {
        t[j] = 0 - O[j];
    }

    Matrix CameraMatrix;
    double **combine = new double *[3];
    combine[0] = u;
    combine[1] = v;
    combine[2] = w;
    
    // fill the matrix top left 3x3 coner
    for (int row=0;row<3;row++) {
        for (int column=0;column<3;column++) {
            CameraMatrix.A[row][column] = combine[column][row];
        }
    }
    // fill the last row and last column
    for (int k=0;k<4;k++) {
        if (k!=3) {
            // first three row last column
            CameraMatrix.A[k][3] = 0;
            // last row first three column
            // dotproduct: use u,v,w after normalize: potentially can cause error 
            CameraMatrix.A[3][k] = VectorDotProduct(combine[k],t);
        }
        // the last row and column, k=3
        else {
            CameraMatrix.A[k][k] = 1;
        }
    }
    return CameraMatrix;
}

Matrix 
Camera::ViewTransform() {
    double a,n,f,cot_a;
    a = angle;
    n = near;
    f = far;
    cot_a = cos(a/2) / sin(a/2);
    // Initialize the matrix with 0
    Matrix ViewMatrix;
    for (int r=0;r<4;r++) {
        for (int c=0;c<4;c++) {
            ViewMatrix.A[r][c] = 0;
        }
    }
    ViewMatrix.A[0][0] = cot_a;
    ViewMatrix.A[1][1] = cot_a;
    ViewMatrix.A[2][2] = (f+n)/(f-n);
    ViewMatrix.A[2][3] = -1;
    ViewMatrix.A[3][2] = (2*f*n)/(f-n);
    return ViewMatrix;
}

Matrix
Camera::DeviceTransform(int n,int m) {
    Matrix DeviceMatrix;
    // initialize the matrix 
    for (int r=0;r<4;r++) {
        for (int c=0;c<4;c++) {
            DeviceMatrix.A[r][c] = 0;
        }
    }
    DeviceMatrix.A[0][0] = n/2;
    DeviceMatrix.A[1][1] = m/2;
    DeviceMatrix.A[2][2] = 1;
    DeviceMatrix.A[3][0] = n/2;
    DeviceMatrix.A[3][1] = m/2;
    DeviceMatrix.A[3][3] = 1;
    return DeviceMatrix;
}

Matrix
TransformTrianglesToDeviceSpace(Camera c,int width,int height) {
    // * denote ComposeMatrices()
    /* generate Camera Transform(4x4): Transform Matrix (slide6-56)(use u,v,w,o to get this matrix, Normalize u,v,w)
    Camera Transform Matrix => M1 -> 
    View Transform (slide6, 62)
    M1 * Transform Matrix => M2 (a,b,c,d) 
    Device Transform (slide6,28)
    M2 * Device Transform => M3 
    // M3 is the total transform(All Matrix, not with vertex), Camera and View Tranform is Matrix, not the result of composition
    (x,y,z,1) * M3 -> Projection (a/d,b/d,c/d) Final result 
    => Draw the triangle beased on Projection result.
    */
    Matrix CameraMatrix = c.CameraTransform();
    Matrix ViewMatrix = c.ViewTransform();
    Matrix DeviceMatrix = c.DeviceTransform(width,height);
    //Compose three Matrix in order: TotalTransform;
    Matrix M1 = M1.ComposeMatrices(CameraMatrix, ViewMatrix);
    Matrix M2 = M2.ComposeMatrices(M1, DeviceMatrix);
    return M2;
}
// -------------------- Finish seting up the matrix for the transformation


// project 1D
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
        sorted_triangle.shading[i] = tri.shading[indexs[i]];
        for (int m=0;m<3;m++) {
            sorted_triangle.colors[i][m] = tri.colors[indexs[i]][m];
            sorted_triangle.normals[i][m] = tri.normals[indexs[i]][m];
        }
    }
    return sorted_triangle;

}

double 
LERP (double t,double fa,double fb) {
    double fx = fa + t*(fb-fa);
    return fx;
}

vector<Triangle>
split_tri(Triangle triangle) {
    // 0:rowMin, 1:split, 2:rowMax
    // find the intersect vertex, in the line connect rowMin and rowMax, so A:rowMin(0), B:rowMax(2)
    double t=0;
    if (triangle.Y[2] != triangle.Y[0]) t = (triangle.Y[1] - triangle.Y[0]) / (triangle.Y[2] - triangle.Y[0]);
    double intersectX,intersectZ,intersectShading;
    //double intersectR,intersectG,intersectB;
    double *intersectRGB = new double [3];
    double *intersectNormals = new double [3]; 
    intersectX = LERP(t,triangle.X[0],triangle.X[2]);
    intersectZ = LERP(t,triangle.Z[0],triangle.Z[2]);
    intersectShading = LERP(t,triangle.shading[0],triangle.shading[2]);

    for (int m=0;m<3;m++) {
        intersectRGB[m] = LERP(t,triangle.colors[0][m],triangle.colors[2][m]);
        intersectNormals[m] = LERP(t,triangle.normals[0][m],triangle.normals[2][m]);
    }

    Triangle uptri,downtri;
    // generate up triangle
    for (int i=1;i<3;i++) {
        //intersectX is on the right side, should be the new split point
        uptri.X[i] = triangle.X[i];
        uptri.Y[i] = triangle.Y[i];
        uptri.Z[i] = triangle.Z[i];
        uptri.shading[i] = triangle.shading[i];
        for (int m=0;m<3;m++) {
            uptri.colors[i][m] = triangle.colors[i][m];
            uptri.normals[i][m] = triangle.normals[i][m];
        }
    }
    uptri.X[0] = intersectX;
    uptri.Y[0] = triangle.Y[1];
    uptri.Z[0] = intersectZ;
    uptri.shading[0] = intersectShading;
    for (int m=0;m<3;m++) {
        uptri.colors[0][m] = intersectRGB[m];
        uptri.normals[0][m] = intersectNormals[m];
    }

    // generate down triangle
    for (int i=0;i<2;i++) {
        //intersectX is on the right side, should be the new split point
        downtri.X[i] = triangle.X[i];
        downtri.Y[i] = triangle.Y[i];
        downtri.Z[i] = triangle.Z[i];
        downtri.shading[i] = triangle.shading[i];
        for (int m=0;m<3;m++) {
            downtri.colors[i][m] = triangle.colors[i][m];
            downtri.normals[i][m] = triangle.normals[i][m];
        }
    }
    downtri.X[2] = intersectX;
    downtri.Y[2] = triangle.Y[1];
    downtri.Z[2] = intersectZ;
    downtri.shading[2] = intersectShading;
    for (int m=0;m<3;m++) {
        downtri.colors[2][m] = intersectRGB[m];
        downtri.normals[2][m] = intersectNormals[m];
    }

    vector<Triangle> UDTriangle (2);
    UDTriangle[0] = uptri;
    UDTriangle[1] = downtri;

    return UDTriangle;
}

void 
fill_up (vtkImageData *image,int width, int height,double *z_buffer,Triangle tri) {
    double rowMin = tri.Y[0];
    double rowMax = tri.Y[2];
    unsigned char *buffer;
    if (floor_441(rowMax) - ceil_441(rowMin) >= 0) { // at least in the same row
        double leftendT = 0;
        double leftendX,leftendZ,leftShading;
        double leftendR,leftendG,leftendB;

        double rightendT = 0;
        double rightendX,rightendZ,rightShading;
        double rightendR,rightendG,rightendB;
        for (int y=ceil_441(rowMin); y<=floor_441(rowMax);y++) {
            if (rowMax != rowMin) 
                leftendT = (y-rowMin) / (rowMax - rowMin);
            leftendX = LERP(leftendT,tri.X[0],tri.X[2]);
            leftendZ = LERP(leftendT,tri.Z[0],tri.Z[2]);
            leftShading = LERP(leftendT,tri.shading[0],tri.shading[2]);
            leftendR = LERP(leftendT,tri.colors[0][0],tri.colors[2][0]);
            leftendG = LERP(leftendT,tri.colors[0][1],tri.colors[2][1]);
            leftendB = LERP(leftendT,tri.colors[0][2],tri.colors[2][2]);

            if (rowMax != tri.Y[1]) 
                rightendT = (y-tri.Y[1]) / (rowMax - tri.Y[1]);
            rightendX = LERP(rightendT,tri.X[1],tri.X[2]);
            rightendZ = LERP(rightendT,tri.Z[1],tri.Z[2]);
            rightShading = LERP(rightendT,tri.shading[1],tri.shading[2]);
            rightendR = LERP(rightendT,tri.colors[1][0],tri.colors[2][0]);
            rightendG = LERP(rightendT,tri.colors[1][1],tri.colors[2][1]);
            rightendB = LERP(rightendT,tri.colors[1][2],tri.colors[2][2]);

            if ( floor_441(rightendX) - ceil_441(leftendX) >= 0) {
                for (int x=ceil_441(leftendX); x<=floor_441(rightendX); x++) {
                    double t = 0;
                    double z,shade;
                    if (rightendX != leftendX) 
                        t = (x-leftendX) / (rightendX - leftendX);
                    z = LERP(t,leftendZ,rightendZ);
                    shade = LERP(t,leftShading,rightShading);
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

                        buffer = (unsigned char *) image->GetScalarPointer(x,y,0);
                        //Color_R = 255*min(1, R*shading_value)
                        buffer[0] = ceil_441(255 * min(1.0,R*shade));
                        buffer[1] = ceil_441(255 * min(1.0,G*shade));
                        buffer[2] = ceil_441(255 * min(1.0,B*shade));
                    }  
                }
            } 
        }
    }
}


void 
fill_down (vtkImageData *image,int width, int height,double *z_buffer,Triangle tri) {
    double rowMin = tri.Y[0];
    double rowMax = tri.Y[2];
    unsigned char *buffer;
    if (floor_441(rowMax) - ceil_441(rowMin) >= 0) { // at least in the same row
        double leftendT = 0;
        double leftendX,leftendZ,leftShading;
        double leftendR,leftendG,leftendB;

        double rightendT = 0;
        double rightendX,rightendZ,rightShading;
        double rightendR,rightendG,rightendB;
        for (int y=ceil_441(rowMin); y<=floor_441(rowMax);y++) {
            if (rowMax != rowMin) 
                leftendT = (y-rowMin) / (rowMax - rowMin);
            leftendX = LERP(leftendT,tri.X[0],tri.X[2]);
            leftendZ = LERP(leftendT,tri.Z[0],tri.Z[2]);
            leftShading = LERP(leftendT,tri.shading[0],tri.shading[2]);
            leftendR = LERP(leftendT,tri.colors[0][0],tri.colors[2][0]);
            leftendG = LERP(leftendT,tri.colors[0][1],tri.colors[2][1]);
            leftendB = LERP(leftendT,tri.colors[0][2],tri.colors[2][2]);

            if (tri.Y[1] != tri.Y[0]) 
                rightendT = (y-tri.Y[0]) / (tri.Y[1] - tri.Y[0]);
            rightendX = LERP(rightendT,tri.X[0],tri.X[1]);
            rightendZ = LERP(rightendT,tri.Z[0],tri.Z[1]);
            rightShading = LERP(rightendT,tri.shading[0],tri.shading[1]);
            rightendR = LERP(rightendT,tri.colors[0][0],tri.colors[1][0]);
            rightendG = LERP(rightendT,tri.colors[0][1],tri.colors[1][1]);
            rightendB = LERP(rightendT,tri.colors[0][2],tri.colors[1][2]);

            if ( floor_441(rightendX) - ceil_441(leftendX) >= 0) {
                for (int x=ceil_441(leftendX); x<=floor_441(rightendX); x++) {
                    double t=0;
                    double z,shade;
                    if (rightendX != leftendX) 
                        t = (x-leftendX) / (rightendX - leftendX);
                    z = LERP(t,leftendZ,rightendZ);
                    shade = LERP(t,leftShading,rightShading);
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
                        buffer = (unsigned char *) image->GetScalarPointer(x,y,0);
                        buffer[0] = ceil_441(255 * min(1.0,R*shade));
                        buffer[1] = ceil_441(255 * min(1.0,G*shade));
                        buffer[2] = ceil_441(255 * min(1.0,B*shade));
                    }  
                }
            }
        }
    }
}
// --------------------------- finish function to split and draw arbitrary/up/down triangle, 

// "main" function call to draw triangle
void 
draw_triangle(vtkImageData *image,Screen screen,Triangle tri) {
    Triangle sort_tri = basic_setup(tri);

    if (sort_tri.Y[0] == sort_tri.Y[1]) {
        fill_up (image,screen.width,screen.height,screen.z_buffer,sort_tri);
    }
    else if (sort_tri.Y[1] == sort_tri.Y[2]) {
        fill_down (image,screen.width,screen.height,screen.z_buffer,sort_tri);
    }
    else {
        vector<Triangle> UDTriangle = split_tri(sort_tri);
        Triangle Up_tri = basic_setup(UDTriangle[0]);
        fill_up (image,screen.width,screen.height,screen.z_buffer,Up_tri);
        Triangle Down_tri = basic_setup(UDTriangle[1]);
        fill_down (image,screen.width,screen.height,screen.z_buffer,Down_tri);
    }  

}


double 
CalculatePhongShading(LightingParameters &lp, double *V, double *N) {
    // V is viewDirection, N is normal, L is lightDir
    /*
    dot means: double VectorDotProduct(double *v1,double *v2)
    Shading_Amount = Ka + Kd*Diffuse + Ks*Specular
    Diffuse = abs(L dot N)
    Specular = cos(α) ^ (alpha)
    cos(a) = max(0,V dot R)
    R = 2*(L dot N)*N - L
    */
    double *L = new double [3];
    for (int i=0;i<3;i++) {
        L[i] = lp.lightDir[i];
    }
    //normalize vector
    Normalize(V);
    Normalize(N);
    Normalize(L);
    //shading setup
    double Diffuse,Specular,Shading_Amount;
    //  calculate diffuse
    Diffuse = abs(VectorDotProduct(L,N));
    //  calculate specular
    double specular_cosA, LdotN;
    double *R = new double [3];
    LdotN = VectorDotProduct(L,N);
    for (int j=0;j<3;j++) {
        R[j] = 2 * LdotN * N[j] - L[j]; // calculate R
    }
    //cout<<"R is "<<R[0]<<", "<<R[1]<<", "<<R[2]<<endl;
    specular_cosA = VectorDotProduct(V,R); // calculate cos(a)
    //cout<<"Cosine between R and viewDir is "<<specular_cosA<<endl;
    Specular = pow(std::max(0.0,specular_cosA),lp.alpha); // calculate Specular

    //combine all three
    Shading_Amount = lp.Ka + lp.Kd*Diffuse + lp.Ks*Specular;

    //cout<<"Shading is "<<Shading_Amount<<", AMB = "<<lp.Ka<<", DIFF = "<<lp.Kd*Diffuse<<", SPEC = "<<lp.Ks*Specular<<endl;
    //cout<<"\n"<<endl;

    return Shading_Amount;
}


// changing triangle's vertex from world space to device space, and call the function to draw triangle
void 
RenderTriangles(Matrix TransformMatrix,vector<Triangle> triangles, vtkImageData *image,Screen screen,Camera camera) {
    // triangles.size()
    for (int t=0;t<triangles.size();t++) {
        //cout <<"Working on triangle "<<t<<"\n"<<endl;

        Triangle triIn = triangles[t];
        Triangle triOut;
        
        // for a triangle, loop through each Vertex
        for (int j=0;j<3;j++) {
            //Function1: calculating shadeing value of the triangle
            // viewDirection = Camera position - Vertex position
            double *viewDirection = new double [3];
            double *normal = new double [3];
            double *three_shadings = new double [3];
            // for a vertex, loop through XYZ coordinate -- Basic setup
            viewDirection[0] = camera.position[0] - triIn.X[j];
            viewDirection[1] = camera.position[1] - triIn.Y[j];
            viewDirection[2] = camera.position[2] - triIn.Z[j];
            for (int m=0;m<3;m++){
                normal[m] = triIn.normals[j][m];
            }
            //cout<<"Working on vertex "<<triIn.X[j]<<", "<<triIn.Y[j]<<", "<<triIn.Z[j]<<endl;
            //cout<<"   Normal is "<<normal[0]<<", "<<normal[1]<<", "<<normal[2]<<endl;
            double one_vertex_shading = CalculatePhongShading(lp,viewDirection,normal);
            three_shadings[j] = one_vertex_shading;

        //Function2: Transfer the vertex from real world to camera position
        // calculate for each vertex of triangle
        //for (int i=0;i<3;i++) {
            double *ptIn = new double [4];
            double *ptOut = new double [4];
            ptIn[0] = triIn.X[j];
            ptIn[1] = triIn.Y[j];
            ptIn[2] = triIn.Z[j];
            ptIn[3] = 1;
            TransformMatrix.TransformPoint(ptIn,ptOut);

            //Function3: Form a new triangle with all the new information
            // project vertex and store to triangle
            triOut.X[j] = ptOut[0] / ptOut[3];
            triOut.Y[j] = ptOut[1] / ptOut[3];
            triOut.Z[j] = ptOut[2] / ptOut[3];
            triOut.shading[j] = three_shadings[j];
            for (int k=0;k<3;k++) {
                triOut.colors[j][k] = triIn.colors[j][k];
                triOut.normals[j][k] = triIn.normals[j][k];
            }
        }
        // draw triangle
        draw_triangle(image,screen,triOut);
    }  
}

int main()
{
    vector<Triangle> triangles = GetTriangles();
    // TODO: For testing, Change this i and t in RenderTriangles(line 776)
    for (int i=0;i<1000;i++) {
        vtkImageData *image = NewImage(1000,1000);
        Screen screen = InitializeScreen(image,1000,1000); //InitializeScreen(image,width,height)
 
        Camera c = GetCamera(i,1000);
        // form the transform matrix for this camera location
        Matrix TransformMatrix = TransformTrianglesToDeviceSpace(c,screen.width,screen.height);
        // Apply the Matrix to all the triangles and draw them
        RenderTriangles(TransformMatrix,triangles,image,screen,c);
        
        std::string s = std::to_string(i);
        s = "frame"+s;
        char const *name = s.c_str();
        WriteImage(image, name);
    }
}