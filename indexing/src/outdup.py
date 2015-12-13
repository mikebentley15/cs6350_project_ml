'''
Contains a simple class for duplicating output: OutDuplicator
'''

class OutDuplicator(object):
    '''
    Simple class for duplicating output to multiple destinations.

    The destinations can be file objects or can be things like
    sys.stdout or sys.stderr.  Basically anything with the
    following methods: write(), writelines(), flush().

    This object also implements those three functions, so it also
    can behave like a file object, when considered for output.

    Attributes:
        files - List of files to duplicate output to
    '''
    def __init__(self, files):
        'Initializes the OutDuplicator'
        self.files = files

    def write(self, message):
        'Outputs the message to all files'
        for fileObj in self.files:
            fileObj.write(message)
            fileObj.flush()

    def writelines(self, lines):
        'Outputs the lines to all files'
        for fileObj in self.files:
            fileObj.writelines(lines)
            fileObj.flush()

    def flush(self):
        'Flushes the output for all files'
        for fileObj in self.files:
            fileObj.flush()

