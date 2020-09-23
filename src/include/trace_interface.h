/* Author: Stephen Pruett and Siavash Zangeneh
 * Date: 9/26/17
 * Description: The interface for defining the branch traces.
 */

#ifndef TRACE_INTERFACE_H
#define TRACE_INTERFACE_H

#include <stdint.h>

enum class BR_TYPE : int8_t {
  NOT_BR          = 0,
  COND_DIRECT     = 1,
  COND_INDIRECT   = 2,
  UNCOND_DIRECT   = 3,
  UNCOND_INDIRECT = 4,
  CALL            = 5,
  RET             = 6,
};

struct HistElt {
  uint64_t pc;
  uint64_t target;
  uint8_t  direction;
  BR_TYPE  type;
} __attribute__((packed));

#endif  // TRACE_INTERFACE_H