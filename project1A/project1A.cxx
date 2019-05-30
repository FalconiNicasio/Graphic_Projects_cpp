#include <iostream>
#include <vtkDataSet.h>
#include <vtkImageData.h>
#include <vtkPNGWriter.h>
#include <vtkPointData.h>

using std::cerr;
using std::endl;

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


int main()
{
   std::cerr << "In main!" << endl;
   vtkImageData *image = NewImage(1024, 1350);
   for (int x=0; x<27; x++) {
      int r = 0;
      if (x/9 == 1){
         r = 128;
      }
      else if (x/9 == 2) {
         r = 255;
      }

      int g = 0;
      if ((x/3)%3 == 1) {
         g = 128;
      }
      else if ((x/3)%3 == 2) {
         g = 255;
      }

      int b = 0;
      if (x%3 == 1) {
         b = 128;
      }
      else if (x%3 == 2) {
         b = 255;
      }
      for (int row=0; row<50; row++) {
         for (int col=0; col<1024; col++) {
            // GetScalarPointer(col,row,height) (x,y,z)
            unsigned char *buffer = (unsigned char *) image->GetScalarPointer(col,50*x+row,0);
            buffer[0] = r;
            buffer[1] = g;
            buffer[2] = b;
         }
      }

   }
   
   
   WriteImage(image, "proj1A");
}
