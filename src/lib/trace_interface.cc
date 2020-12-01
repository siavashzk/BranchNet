#include "trace_interface.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>

std::vector<HistElt> read_trace(char* input_trace, int max_brs) {
  std::vector<HistElt> history;

  const int BUFFER_SIZE = 4096;
  char      cmd[BUFFER_SIZE];

  auto out = snprintf(cmd, BUFFER_SIZE, "bzip2 -dc %s", input_trace);
  assert(out < BUFFER_SIZE);

  FILE*   fptr = popen(cmd, "r");
  HistElt history_elt_buffer;
  for(int i = 0; i < max_brs && fread(&history_elt_buffer,
                                      sizeof(history_elt_buffer), 1, fptr) == 1;
      ++i) {
    history.push_back(history_elt_buffer);
  }

  if(max_brs == std::numeric_limits<int>::max() && !feof(fptr)) {
    std::cerr << "Error while reading the input trace file\n";
    std::exit(1);
  }
  pclose(fptr);

  return history;
}
