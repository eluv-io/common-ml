    
from loguru import logger

from common_ml.tagging.messages import *
from common_ml.tagging.abstract import FileTagger, MessageProducer


def get_message_producer_from_file_tagger(file_tagger: FileTagger, continue_on_error: bool = False) -> MessageProducer:
    
    class NewMessageProducer(MessageProducer):
        def produce_messages(self, files: List[str]) -> List[Message]:
            res = []

            for fname in files:
                try:
                    file_tagger.tag(fname)
                except Exception as e:
                    logger.opt(exception=e).error(f"Error processing file {fname}")
                    res.append(ErrorMessage(type='error', data=Error(message=str(e), source_media=fname)))
                    if not continue_on_error:
                        return res
                # finished fname, add progress
                res.append(ProgressMessage(type='progress', data=Progress(source_media=fname)))

            return res

    return NewMessageProducer()