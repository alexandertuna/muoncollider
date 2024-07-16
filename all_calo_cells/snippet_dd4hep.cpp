//
// cf: https://github.com/AIDASoft/DD4hep/issues/1297
// Run like:
// root -l -b -q snippet_dd4hep.cpp
//

#include "DD4hep/Detector.h"
#include "DD4hep/DD4hepUnits.h"
#include "DD4hep/BitFieldCoder.h"

R__LOAD_LIBRARY(libDDCore)

int snippet_dd4hep() {

  std::cout << "In the main loop" << std::endl;
  
  auto lcdd = &(dd4hep::Detector::getInstance());
  lcdd->fromCompact("/scratch/tuna/detector-simulation/geometries/MuColl_10TeV_v0A/MuColl_10TeV_v0A.xml");

  // map of detectors, geometry
  for( auto & [name, det] : lcdd->detectors() ) {
    std::cout << "detector: " << name.c_str() << std::endl;
    if (name == "ECalEndcap") {
      std::cout << det.access() << std::endl;
    }
  }

  // map of sensitive detectors, readout
  for( auto & [name, sd] : lcdd->sensitiveDetectors() ) {
    std::cout << "sensitive detector: " << name.c_str() << " " << sd.access() << std::endl;
    // const auto& ro = ()(sd)->readout();
  }

  // playing
  const auto& de = lcdd->detector("ECalEndcap");
  std::cout << "type: " << de.type() << std::endl;
  // const auto& ro = de.readout();
  
  return 0;
}
