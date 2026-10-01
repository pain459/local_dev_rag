import pytest

from local_dev_rag.domain import InvalidRequestError, RequestIdentity


def test_identity_uses_supplied_project_id_even_if_root_changes():
    identity = RequestIdentity.from_headers(
        {
            "X-OpenCode-Session-ID": "session",
            "X-OpenCode-Project-ID": "stable-id",
            "X-OpenCode-Project-Root": "/another/checkout",
        }
    )
    assert identity.session_id == "session"
    assert identity.project_id == "stable-id"
    assert identity.project_root == "/another/checkout"


@pytest.mark.parametrize("header", ["x-opencode-session-id", "x-opencode-project-id"])
@pytest.mark.parametrize("value", [None, "", "   "])
def test_missing_or_empty_identity_header_is_rejected(header, value):
    headers = {"x-opencode-session-id": "session", "x-opencode-project-id": "project"}
    if value is None:
        del headers[header]
    else:
        headers[header] = value
    with pytest.raises(InvalidRequestError) as error:
        RequestIdentity.from_headers(headers)
    assert error.value.param == header


def test_project_root_is_optional_diagnostic_metadata():
    identity = RequestIdentity.from_headers(
        {
            "x-opencode-session-id": "session",
            "x-opencode-project-id": "project",
        }
    )
    assert identity.project_root is None
