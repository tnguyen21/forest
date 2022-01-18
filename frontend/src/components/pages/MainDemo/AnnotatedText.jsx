import { React, useState } from "react";
import styled from "styled-components";

import { Box } from "@mui/material";

const AnnotatedTextContainer = styled.div`
  font-family: "Roboto", sans-serif;
  line-height: 1.5;
`;

function AnnotatedText({ outputText, annotations }) {
  // TODO some logic should be in place to style annotations

  const getHighlightedText = (text, highlight) => {
    // Split text on highlight term, include term itself into parts, ignore case
    const parts = text.split(new RegExp(`(${highlight})`, "gi"));
    return (
      <AnnotatedTextContainer>
        {parts.map((part) =>
          part.toLowerCase() === highlight.toLowerCase() ? <b>{part}</b> : part
        )}
      </AnnotatedTextContainer>
    );
  };

  return (
    <Box sx={{ width: "100%", height: 150 }}>
      {getHighlightedText(outputText, "Freud")}
    </Box>
  );
}

export default AnnotatedText;
