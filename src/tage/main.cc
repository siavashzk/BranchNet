#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <unordered_map>
#include <vector>

#include "cbp2016_utils.h"
#include "trace_interface.h"

#define TAGESCL_64 1
#define TAGESCL_56 2
#define MTAGE_SC 3
#define GTAGE_SC 4
#define GTAGE_SC_NOLOCAL 5
#define GTAGE 6

#ifndef PREDICTOR_CONFIG
#define PREDICTOR_CONFIG TAGESCL_64
#endif

#if PREDICTOR_CONFIG == TAGESCL_64
#define PREDICTOR_SIZE 64
#include "tagescl.h"

#elif PREDICTOR_CONFIG == TAGESCL_56
#define PREDICTOR_SIZE 56
#include "tagescl.h"

#elif PREDICTOR_CONFIG == MTAGE_SC
#include "mtagesc.h"

#elif PREDICTOR_CONFIG == GTAGE_SC
#define ONLY_GTAGE
#include "mtagesc.h"

#elif PREDICTOR_CONFIG == GTAGE_SC_NOLOCAL
#define ONLY_GTAGE
#define DISABLE_LOCAL
#include "mtagesc.h"

#elif PREDICTOR_CONFIG == GTAGE
#define ONLY_GTAGE
#define DISABLE_SC
#define DISABLE_LOCAL
#include "mtagesc.h"

#else
static_assert(false, "Config name not supported");
#endif

OpType convert_brtype_to_optype(BR_TYPE br_type) {
  switch(br_type) {
    case BR_TYPE::NOT_BR:
      return OPTYPE_OP;
    case BR_TYPE::COND_DIRECT:
      return OPTYPE_JMP_DIRECT_COND;
    case BR_TYPE::COND_INDIRECT:
      return OPTYPE_JMP_INDIRECT_COND;
    case BR_TYPE::UNCOND_DIRECT:
      return OPTYPE_JMP_DIRECT_UNCOND;
    case BR_TYPE::UNCOND_INDIRECT:
      return OPTYPE_JMP_INDIRECT_UNCOND;
    case BR_TYPE::CALL:
      return OPTYPE_CALL_DIRECT_UNCOND;
    case BR_TYPE::RET:
      return OPTYPE_RET_UNCOND;
    default:
      std::cerr << "Unexpected BR_TYPE: " << static_cast<int8_t>(br_type)
                << '\n';
      std::exit(1);
  }
}

struct Args {
  char* input_trace_path;
  char* output_file_path;
  char* hard_br_file_path;
};
Args parse_args(int argc, char** argv) {
  if(argc < 2 || argc > 4) {
    std::cerr << "Usage: " << argv[0]
              << " input_trace output_path [hard_br_file_path]\n";
    std::exit(1);
  }

  if(argc == 3)
    return {argv[1], argv[2], nullptr};
  else
    return {argv[1], argv[2], argv[3]};
}

std::vector<HistElt> read_trace(char* input_trace) {
  std::vector<HistElt> history;

  const int BUFFER_SIZE = 4096;
  char      cmd[BUFFER_SIZE];

  auto out = snprintf(cmd, BUFFER_SIZE, "bzip2 -dc %s", input_trace);
  assert(out < BUFFER_SIZE);

  FILE*   fptr = popen(cmd, "r");
  HistElt history_elt_buffer;
  while(fread(&history_elt_buffer, sizeof(history_elt_buffer), 1, fptr) == 1) {
    history.push_back(history_elt_buffer);
  }

  if(!feof(fptr)) {
    std::cerr << "Error while reading the input trace file\n";
    std::exit(1);
  }

  return history;
}

class Stats {
 public:
  void update(uint64_t pc, bool pred, bool dir) {
    update_record(&aggregate_record, pred, dir);
    update_record(&records_map[pc], pred, dir);
  }

  void dump(const char* output_path) {
    const auto records = get_sorted_br_records();


    std::ofstream ofs(output_path);
    ofs << "Branch PC,Accuracy,Mispredictions,Correct "
           "Predictions,Total,dir_t_pred_t,dir_t_pred_nt,dir_nt_pred_t,dir_nt_"
           "pred_nt\n";
    auto print_record = [&ofs](uint64_t pc, const Record& record) {
      if(pc == 0) {
        ofs << "aggregate";
      } else {
        ofs << "0x" << std::hex << pc;
      }
      ofs << ',' << std::dec << 100.0 * record.correct() / record.total()
          << "%," << record.incorrect() << ',' << record.correct() << ','
          << record.total() << ',' << record.dir_t_pred_t << ','
          << record.dir_t_pred_nt << ',' << record.dir_nt_pred_t << ','
          << record.dir_nt_pred_nt << '\n';
    };

    print_record(0, aggregate_record);
    for(auto & [ pc, record ] : records) {
      assert(pc != 0);
      print_record(pc, record);
    }
  }

 private:
  struct Record {
    int64_t dir_t_pred_t   = 0;
    int64_t dir_t_pred_nt  = 0;
    int64_t dir_nt_pred_t  = 0;
    int64_t dir_nt_pred_nt = 0;

    int64_t correct() const { return dir_t_pred_t + dir_nt_pred_nt; }
    int64_t incorrect() const { return dir_t_pred_nt + dir_nt_pred_t; }
    int64_t total() const { return correct() + incorrect(); }
  };

  struct Br_Record {
    uint64_t pc;
    Record   record;
  };

  void update_record(Record* record, bool pred, bool dir) {
    if(dir && pred) {
      record->dir_t_pred_t += 1;
    } else if(dir && !pred) {
      record->dir_t_pred_nt += 1;
    } else if(!dir && pred) {
      record->dir_nt_pred_t += 1;
    } else {
      record->dir_nt_pred_nt += 1;
    }
  }

  std::vector<Br_Record> get_sorted_br_records() {
    std::vector<Br_Record> records;
    records.reserve(records_map.size());
    for(auto & [ pc, record ] : records_map) {
      records.push_back({pc, record});
    }

    std::sort(records.begin(), records.end(), [](auto& lhs, auto& rhs) {
      return lhs.record.incorrect() > rhs.record.incorrect();
    });

    return records;
  };

  Record                               aggregate_record;
  std::unordered_map<uint64_t, Record> records_map;
};

int main(int argc, char** argv) {
  const auto args     = parse_args(argc, argv);
  const auto br_trace = read_trace(args.input_trace_path);

  auto  predictor = std::make_unique<PREDICTOR>(args.hard_br_file_path);
  Stats stats;

  for(const auto& br : br_trace) {
    if(br.type == BR_TYPE::COND_DIRECT || br.type == BR_TYPE::COND_INDIRECT) {
      const bool pred = predictor->GetPrediction(br.pc);
      predictor->UpdatePredictor(br.pc, convert_brtype_to_optype(br.type),
                                 br.direction, pred, br.target);
      stats.update(br.pc, pred, br.direction);
    } else {
      predictor->TrackOtherInst(br.pc, convert_brtype_to_optype(br.type),
                                br.direction, br.target);
    }
  }

  stats.dump(args.output_file_path);
}