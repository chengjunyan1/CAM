This folder contains the physical simulator stuffs


How to use:

You need to firstly modify this in your code (example in simulator_constraint.cpp):

void specialFunction(int key, int x, int y)
{
    time_t rawtime;
    struct tm * timeinfo;
    char buffer[80];

    time (&rawtime);
    timeinfo = localtime(&rawtime);

    strftime(buffer,sizeof(buffer),"%d-%m-%Y %H:%M:%S",timeinfo);
    std::string timenow(buffer);
    std::string prefix="/home/acjy777/chengjunyan1/Robotics/Wrinkles/fem/outputs/sheet_";
    std::string postfix=".obj";
    std::string save_dir=prefix+timenow+postfix;

  switch (key)
  {
    ...
    case GLUT_KEY_HOME: // Press HOME to save state
        sheet->save(save_dir.c_str());
      break;
    ...
  }
}

In Blender, edit texture:

1. Rotate x by 90d in shader editor

2. Use generated coord


Run simulator:

1. You firstly need to move autogen.sh to catkin_ws/

2. Edit visualgen.py to design the movement, you can preview it by running it

3. Edit autogen.sh to desinate your 

  basedir: where the visualgen.py lies on 

  ifile: a base file which containing the code generation marks, you can directly use simulator_constraint.cpp as your ifile

  ofile: the generated code, it should be the script for your simulator (should consistent with your ros launch file)

4. Press HOME to capture the state, it should output to output/ under your basedir

5. Run preset.blend, you can either import the captured state or directly run it which will import latest captured state automatically, and render it