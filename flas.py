from flask import Flask

from bottle import static_file, route, run, redirect, get, post, request
import classifier


@route('/<filename>')
def server_static(filename):
    return static_file(filename, root='./static')


@route('/css/<filename>')
def server_static(filename):
    return static_file(filename, root='./static/css')


@route('/fonts/<filename>')
def server_static(filename):
    return static_file(filename, root='./static/fonts')


@route('/js/<filename>')
def server_static(filename):
    return static_file(filename, root='./static/js')


@get('/')
def index_get():
    redirect("/index.html")


@post('/')
def index_post():
    text = request.forms.text
    return classifier.predictor(str(text))


