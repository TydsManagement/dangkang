import base64
import click

from flask import Flask
from werkzeug.security import generate_password_hash

from api.db.services import UserService


@click.command('reset-password', help='Reset the account password.')
@click.option('--email', prompt=True, help='The email address of the account whose password you need to reset')
@click.option('--new-password', prompt=True, help='the new password.')
@click.option('--password-confirm', prompt=True, help='the new password confirm.')
def reset_password(email, new_password, password_confirm):
    if str(new_password).strip() != str(password_confirm).strip():
        click.echo(click.style('sorry. The two passwords do not match.', fg='red'))
        return
    user = UserService.query(email=email)
    if not user:
        click.echo(click.style('sorry. The Email is not registered!.', fg='red'))
        return
    encode_password = base64.b64encode(new_password.encode('utf-8')).decode('utf-8')
    password_hash = generate_password_hash(encode_password)
    user_dict = {
        'password': password_hash
    }
    UserService.update_user(user[0].id,user_dict)
    click.echo(click.style('Congratulations! Password has been reset.', fg='green'))


def register_commands(app: Flask):
    app.cli.add_command(reset_password)