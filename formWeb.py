from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField, MultipleFileField


class UploadForm(FlaskForm):
    photo = MultipleFileField('Upload Image')
    submit = SubmitField()