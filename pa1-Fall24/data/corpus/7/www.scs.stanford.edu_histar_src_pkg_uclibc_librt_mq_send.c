mq_send c functions for sending to message queue include errno h include stddef h include sys syscall h include mqueue h librt_hidden_proto mq_timedsend ifdef __nr_mq_timedsend define __nr___syscall_mq_timedsend __nr_mq_timedsend static inline _syscall5 int __syscall_mq_timedsend int mqdes const char msg_ptr size_t msg_len unsigned int msg_prio const void abs_timeout endif add a message to queue if o_nonblock is set and queue is full wait for sufficient room in the queue until abs_timeout expires int mq_timedsend mqd_t mqdes const char msg_ptr size_t msg_len unsigned int msg_prio const struct timespec abs_timeout ifdef __nr_mq_timedsend return __syscall_mq_timedsend mqdes msg_ptr msg_len msg_prio abs_timeout else errno enosys return 1 endif librt_hidden_def mq_timedsend add a message to queue int mq_send mqd_t mqdes const char msg_ptr size_t msg_len unsigned int msg_prio return mq_timedsend mqdes msg_ptr msg_len msg_prio null
