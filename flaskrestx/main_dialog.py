from flask import Flask
from flask_restx import Api, Resource, fields
from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)
api = Api(
    app,
    version="1.0",
    title="toyouchat API",
    description="A simple toyouCHAT API",
)

ns = api.namespace("dialogs", description="CHAT operations")

dialog = api.model(
    "dialog",
    {
        "id": fields.Integer(readonly=True, description="The dialog unique identifier"),
        "dialog": fields.String(required=True, description="The dialog details"),
    },
)


class dialogDAO(object):
    def __init__(self):
        self.counter = 0
        self.dialogs = []

    def get(self, id):
        for dialog in self.dialogs:
            if dialog["id"] == id:
                return dialog
        api.abort(404, "dialog {} doesn't exist".format(id))

    def create(self, data):
        dialog = data
        dialog["id"] = self.counter = self.counter + 1
        self.dialogs.append(dialog)
        return dialog

    def update(self, id, data):
        dialog = self.get(id)
        dialog.update(data)
        return dialog

    def delete(self, id):
        dialog = self.get(id)
        self.dialogs.remove(dialog)


dialog_dao = dialogDAO()
dialog_dao.create({"dialog": "Build an API"})
dialog_dao.create({"dialog": "?????"})
dialog_dao.create({"dialog": "profit!"})


@ns.route("/")
class dialogList(Resource):
    """Shows a list of all dialogs, and lets you POST to add new dialogs"""

    @ns.doc("list_dialogs")
    @ns.marshal_list_with(dialog)
    def get(self):
        """List all dialogs"""
        return dialog_dao.dialogs

    @ns.doc("create_dialog")
    @ns.expect(dialog)
    @ns.marshal_with(dialog, code=201)
    def post(self):
        """Create a new dialog"""
        return dialog_dao.create(api.payload), 201


@ns.route("/<int:id>")
@ns.response(404, "dialog not found")
@ns.param("id", "The dialog identifier")
class dialog(Resource):
    """Show a single dialog item and lets you delete them"""

    @ns.doc("get_dialog")
    @ns.marshal_with(dialog)
    def get(self, id):
        """Fetch a given resource"""
        return dialog_dao.get(id)

    @ns.doc("delete_dialog")
    @ns.response(204, "dialog deleted")
    def delete(self, id):
        """Delete a dialog given its identifier"""
        dialog_dao.delete(id)
        return "", 204

    @ns.expect(dialog)
    @ns.marshal_with(dialog)
    def put(self, id):
        """Update a dialog given its identifier"""
        return dialog_dao.update(id, api.payload)


if __name__ == "__main__":
    app.run(debug=True)
