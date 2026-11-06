"""Same actual standalone dispatcher, exposing its cache to the barrier."""
import heapq,logging
from federatedscope.core.fed_runner import StandaloneRunner
logger=logging.getLogger(__name__)
class BoundaryRunner(StandaloneRunner):
    def _run_simulation(self):
        """
        Run for standalone simulation (W/O online aggr)
        """
        server_msg_cache = list()
        self.server._pending_server_messages = server_msg_cache
        while True:
            if len(self.shared_comm_queue) > 0:
                msg = self.shared_comm_queue.popleft()
                if not self.cfg.vertical.use and msg.receiver == [
                        self.server_id
                ]:
                    # For the server, move the received message to a
                    # cache for reordering the messages according to
                    # the timestamps
                    msg.serial_num = self.serial_num_for_msg
                    self.serial_num_for_msg += 1
                    heapq.heappush(server_msg_cache, msg)
                else:
                    self._handle_msg(msg)
            elif len(server_msg_cache) > 0:
                msg = heapq.heappop(server_msg_cache)
                if self.cfg.asyn.use and self.cfg.asyn.aggregator \
                        == 'time_up':
                    # When the timestamp of the received message beyond
                    # the deadline for the currency round, trigger the
                    # time up event first and push the message back to
                    # the cache
                    if self.server.trigger_for_time_up(msg.timestamp):
                        heapq.heappush(server_msg_cache, msg)
                    else:
                        self._handle_msg(msg)
                else:
                    self._handle_msg(msg)
            else:
                if self.cfg.asyn.use and self.cfg.asyn.aggregator \
                        == 'time_up':
                    self.server.trigger_for_time_up()
                    if len(self.shared_comm_queue) == 0 and \
                            len(server_msg_cache) == 0:
                        break
                else:
                    # terminate when shared_comm_queue and
                    # server_msg_cache are all empty
                    break
