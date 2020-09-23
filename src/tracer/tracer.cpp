/* Authors: Stephen Pruett, Siavash Zangeneh
 * Description: This pin tools creates a trace of dynamic branches.
 */

#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include <memory>

#include "control_manager.H"
#include "instlib.H"
#include "pin.H"
#include "pinplay.H"

#include "trace_interface.h"

using namespace INSTLIB;
using namespace CONTROLLER;

PINPLAY_ENGINE pinplay_engine;
KNOB<BOOL>     KnobPinPlayLogger(KNOB_MODE_WRITEONCE, "pintool", "log", "0",
                             "Activate the pinplay logger");
KNOB<BOOL> KnobPinPlayReplayer(KNOB_MODE_WRITEONCE, "pintool", "replay", "0",
                               "Activate the pinplay replayer");
CONTROL_MANAGER controller_engine("controller_");

/* ===================================================================== */
/* Commandline Switches */
/* ===================================================================== */
KNOB<string> KnobTraceFile(KNOB_MODE_WRITEONCE, "pintool", "trace_out_file", "",
                           "specify branch trace output file name");
KNOB<string> KnobCompressorPath(KNOB_MODE_WRITEONCE, "pintool", "compressor",
                                "/usr/bin/bzip2", "Path to compressor program");
KNOB<int64_t> KnobWarmupInstructions(
  KNOB_MODE_WRITEONCE, "pintool", "warmup_instructions", "20000000",
  "Number of warmup instructions before tracing and/or branch predicting.");
KNOB<int64_t> KnobRedirectPCAtStart(
  KNOB_MODE_WRITEONCE, "pintool", "redirect_pc_at_start", "0",
  "If not zero, redirect the program to this PC.");

/* ===================================================================== */
/* Print Help Message                                                    */
/* ===================================================================== */
int32_t Usage() {
  cerr << "This pin tool generates a dynamic branch trace.\n\n";
  cerr << KNOB_BASE::StringKnobSummary();
  cerr << endl;
  return -1;
}

/* ===================================================================== */
/* Static Global Variables */
/* ===================================================================== */
namespace {

bool  started = false;
FILE* trace_file;

BR_TYPE get_br_type(const INS& inst) {
  switch(INS_Category(inst)) {
    case XED_CATEGORY_COND_BR:
      return INS_IsDirectBranch(inst) ? BR_TYPE::COND_DIRECT :
                                        BR_TYPE::COND_INDIRECT;
    case XED_CATEGORY_UNCOND_BR:
      return INS_IsDirectBranch(inst) ? BR_TYPE::UNCOND_DIRECT :
                                        BR_TYPE::UNCOND_INDIRECT;
    case XED_CATEGORY_CALL:
      return BR_TYPE::CALL;
    case XED_CATEGORY_RET:
      return BR_TYPE::RET;
    default:
      return BR_TYPE::NOT_BR;
  }
}

}  // namespace

void dump_br(const ADDRINT fetch_addr, const BOOL resolve_dir,
             const ADDRINT branch_target, UINT32 br_type) {
  if(!started)
    return;

  HistElt current_hist_elt;
  current_hist_elt.pc        = fetch_addr;
  current_hist_elt.target    = branch_target;
  current_hist_elt.direction = resolve_dir ? 1 : 0;
  current_hist_elt.type      = static_cast<BR_TYPE>(br_type);

  static_assert(sizeof(ADDRINT) == sizeof(current_hist_elt.pc));

  auto elements_written = fwrite(&current_hist_elt, sizeof(current_hist_elt), 1,
                                 trace_file);
  assert(elements_written == 1);
}

VOID redirect_to_pc(CONTEXT* ctx) {
  started     = true;
  ADDRINT rip = KnobRedirectPCAtStart.Value();
  PIN_SetContextRegval(ctx, REG_INST_PTR, (const UINT8*)(&rip));
  PIN_RemoveInstrumentation();
  PIN_ExecuteAt(ctx);
}

VOID instrumentation_function(INS inst, VOID* v) {
  if(KnobRedirectPCAtStart.Value() != 0 && !started) {
    INS_InsertCall(inst, IPOINT_BEFORE, (AFUNPTR)redirect_to_pc, IARG_CONTEXT,
                   IARG_END);
    return;
  }

  BR_TYPE br_type = get_br_type(inst);

  if(br_type != BR_TYPE::NOT_BR) {
    INS_InsertCall(inst, IPOINT_BEFORE, (AFUNPTR)dump_br, IARG_INST_PTR,
                   IARG_BRANCH_TAKEN, IARG_BRANCH_TARGET_ADDR, IARG_UINT32,
                   static_cast<uint32_t>(br_type), IARG_END);
  }
}

/* ===================================================================== */

VOID fini_function(int, VOID* v) {
  fclose(trace_file);
}

void control_handler(EVENT_TYPE ev, VOID* val, CONTEXT* ctxt, VOID* ip,
                     THREADID tid, BOOL bcast) {
  switch(ev) {
    case EVENT_START:
      started = true;
      break;

    case EVENT_STOP:
      PIN_ExitApplication(0);
      break;

    default:
      ASSERTX(false);
      break;
  }
}

/* ===================================================================== */
/* Main                                                                  */
/* ===================================================================== */

int main(int argc, CHAR* argv[]) {
  PIN_InitSymbols();
  if(PIN_Init(argc, argv)) {
    return Usage();
  }

  pinplay_engine.Activate(argc, argv, KnobPinPlayLogger, KnobPinPlayReplayer);

  controller_engine.RegisterHandler(control_handler, 0, false);
  controller_engine.Activate();

  INS_AddInstrumentFunction(instrumentation_function, 0);
  PIN_AddFiniFunction(fini_function, 0);

  if(KnobCompressorPath.Value().empty()) {
    trace_file = fopen(KnobTraceFile.Value().c_str(), "w");
  } else {
    char bzip2_pipe_cmd[1024];
    sprintf(bzip2_pipe_cmd, "%s > %s", KnobCompressorPath.Value().c_str(),
            KnobTraceFile.Value().c_str());
    trace_file = popen(bzip2_pipe_cmd, "w");
  }

  // Never returns
  PIN_StartProgram();

  return 0;
}