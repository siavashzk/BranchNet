#include "utils.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>

void Stats::update(uint64_t pc, bool pred, bool dir) {
  update_record(&aggregate_record, pred, dir);
  update_record(&records_map[pc], pred, dir);
}

void Stats::dump(const char* output_path) {
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
    ofs << ',' << std::dec << 100.0 * record.correct() / record.total() << "%,"
        << record.incorrect() << ',' << record.correct() << ','
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

void Stats::print_br_stats(uint64_t pc) {
  const auto& record = records_map[pc];
  std::cout << "Accuracy: " << std::dec
            << 100.0 * record.correct() / record.total()
            << "%, total: " << record.total() << ", Breakdown:"
            << record.dir_t_pred_t << ',' << record.dir_t_pred_nt << ','
            << record.dir_nt_pred_t << ',' << record.dir_nt_pred_nt << '\n';
}

void Stats::update_record(Record* record, bool pred, bool dir) {
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

std::vector<Stats::Br_Record> Stats::get_sorted_br_records() {
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