#ifndef LIB_UTILS_H
#define LIB_UTILS_H

#include <cstdint>
#include <unordered_map>
#include <vector>

class Stats {
 public:
  void update(uint64_t pc, bool pred, bool dir);
  void dump(const char* output_path);
  void print_br_stats(uint64_t pc);

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

  void                   update_record(Record* record, bool pred, bool dir);
  std::vector<Br_Record> get_sorted_br_records();

  Record                               aggregate_record;
  std::unordered_map<uint64_t, Record> records_map;
};

#endif  // LIB_UTILS_H