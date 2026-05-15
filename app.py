#!/usr/bin/env python3
"""Streamlit entry point for the Funnel Intelligence RAG Platform."""

import streamlit as st

from frameworks.streamlit_app import render_app

if __name__ == "__main__":
    render_app()
